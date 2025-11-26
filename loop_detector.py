"""
Loop Detection and Extraction Node for ComfyUI
自动检测视频中的循环模式并提取循环片段及其前后帧
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class LoopDetectExtract:
    """
    检测视频中的循环模式并提取循环片段
    
    使用 FFT 自相关分析检测周期性，然后精确定位循环边界
    """
    
    DESCRIPTION = "自动检测视频帧序列中的循环模式，提取循环片段。使用 GPU 加速的 FFT 自相关分析和相似度匹配。"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {
                        "tooltip": "输入图像序列 (batch, height, width, channels)，值范围 [0, 1]"
                    }
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "循环检测的置信度阈值，低于此值返回全部帧"
                    }
                ),
            },
            "optional": {
                "min_period": (
                    "INT",
                    {
                        "default": 24,
                        "min": 2,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "最小循环周期（帧数）"
                    }
                ),
                "max_period": (
                    "INT",
                    {
                        "default": 300,
                        "min": 4,
                        "max": 5000,
                        "step": 1,
                        "tooltip": "最大循环周期（帧数）"
                    }
                ),
                "analysis_stride": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "分析时的帧采样步长，越大越快但精度降低"
                    }
                ),
                "analysis_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 512,
                        "step": 32,
                        "tooltip": "分析时图像缩放的目标尺寸"
                    }
                ),
                "seam_threshold": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "首尾接缝相似度阈值，超过则优先判定整段闭环"
                    }
                ),
                "min_pairs": (
                    "INT",
                    {
                        "default": 12,
                        "min": 4,
                        "max": 200,
                        "step": 1,
                        "tooltip": "候选周期需满足的最小成对样本数"
                    }
                ),
                "prefer_longer_cycles": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "在分数接近时倾向选择更长周期"
                    }
                ),
                "length_bias": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.01,
                        "tooltip": "对较长周期的轻度偏好强度"
                    }
                ),
                "motion_weight": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "将运动周期性引入候选评分的权重"
                    }
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "loop_frames",
        "loop_start",
        "loop_period",
        "confidence",
        "report"
    )
    FUNCTION = "detect_and_extract"
    CATEGORY = "kivi_nodes"
    
    def detect_and_extract(
        self,
        frames: torch.Tensor,
        confidence_threshold: float,
        min_period: int = 24,
        max_period: int = 300,
        analysis_stride: int = 2,
        analysis_size: int = 256,
        seam_threshold: float = 0.85,
        min_pairs: int = 12,
        prefer_longer_cycles: bool = True,
        length_bias: float = 0.05,
        motion_weight: float = 0.25,
    ) -> Tuple:
        """
        检测循环并提取循环帧
        
        Returns:
            (loop_frames, loop_start, loop_period, confidence, report)
        """
        device = self._get_device(frames)
        N = frames.shape[0]
        
        # 边界检查
        if N < min_period + 1:
            return self._return_all_frames(frames, f"帧数太少 ({N} < {min_period+1})")
        
        try:
            # 1. 预处理：采样和缩放
            frames_analysis = self._prepare_for_analysis(
                frames, stride=analysis_stride, target_size=analysis_size, device=device
            )
            
            # 2. 特征提取
            features = self._extract_features(frames_analysis, device=device)
            
            # 2.1 下采样灰度（用于 seam 和运动分析）
            gray_small = self._downscaled_gray(frames, device=device, size=64)
            
            # 2.2 接缝预检（整段闭环检测）
            seam_sim = self._seam_score(gray_small, window=3)
            
            # 2.3 运动周期性（用于候选评分的加权）
            motion_ac = self._motion_autocorr(gray_small)  # (N-1,)
            
            # 3. FFT 自相关分析
            autocorr = self._compute_autocorrelation(features)
            
            # 4. 检测循环周期（使用增强的多因素评分）
            period_stride, confidence, candidates = self._detect_period(
                autocorr,
                N=len(frames_analysis),  # 传入分析帧数用于样本对惩罚
                min_period=max(1, min_period // analysis_stride),
                max_period=min(len(autocorr) - 1, max_period // analysis_stride),
                motion_ac=motion_ac[:len(autocorr)] if motion_ac is not None else None,
                min_pairs=min_pairs,
                prefer_longer=prefer_longer_cycles,
                length_bias=length_bias,
                motion_weight=motion_weight,
            )
            
            # 映射回原始帧空间
            period = period_stride * analysis_stride
            
            # 4.5 优先处理"整段闭环"场景（seam 预检）
            if confidence < confidence_threshold and seam_sim >= seam_threshold:
                # 内部没有足够强峰，但首尾接缝相似度很高 → 判定整段为一个循环
                loop_start = 0
                period = N
                loop_frames = self._extract_frames(frames, loop_start, period)
                report = self._generate_report(
                    N, loop_start, period, float(seam_sim),
                    [{"period": N, "score": float(seam_sim), "prominence": float(seam_sim)}]
                )
                report = f"[整段闭环检测] 接缝相似度: {seam_sim:.3f}\n" + report
                return (
                    loop_frames,
                    0, int(period), float(seam_sim), report
                )
            
            # 5. 检查置信度
            if confidence < confidence_threshold:
                return self._return_all_frames(
                    frames, 
                    f"置信度不足 ({confidence:.3f} < {confidence_threshold})"
                )
            
            # 6. 相位锁定：找到最佳起始点
            loop_start = self._find_best_start(frames, period, device=device)
            
            # 7. 边界精修
            loop_start, period = self._refine_boundaries(
                frames, loop_start, period, device=device
            )
            
            # 8. 提取循环帧
            loop_frames = self._extract_frames(frames, loop_start, period)
            
            # 9. 生成报告
            report = self._generate_report(
                N, loop_start, period, confidence, candidates
            )
            
            return (
                loop_frames,
                int(loop_start),
                int(period),
                float(confidence),
                report
            )
            
        except Exception as e:
            print(f"[LoopDetectExtract] 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._return_all_frames(frames, f"处理出错: {str(e)}")
        finally:
            # 清理 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _get_device(self, tensor: torch.Tensor) -> torch.device:
        """获取合适的计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return tensor.device
    
    def _downscaled_gray(self, frames: torch.Tensor, device: torch.device, size: int = 64) -> torch.Tensor:
        """
        下采样灰度图像（用于 seam 检测和运动分析）
        
        Args:
            frames: 输入帧 (N, H, W, C)
            device: 计算设备
            size: 目标尺寸
            
        Returns:
            灰度图像 (N, size, size)
        """
        frames = frames.to(device)
        if frames.shape[-1] >= 3:
            gray = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2])
        else:
            gray = frames[..., 0]
        
        gray_small = F.interpolate(
            gray.unsqueeze(1), 
            size=(size, size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        
        return gray_small  # (N, size, size)
    
    def _seam_score(self, gray_small: torch.Tensor, window: int = 3) -> float:
        """
        计算首尾接缝相似度（用于检测整段循环）
        
        Args:
            gray_small: 灰度图像 (N, H, W)
            window: 比较的帧窗口大小
            
        Returns:
            相似度分数 [0, 1]，越高表示首尾越相似
        """
        N = gray_small.shape[0]
        w = min(window, N // 2)
        if w <= 0:
            return 0.0
        
        # 比较最后 w 帧和最前 w 帧
        last_frames = gray_small[N - w: N]   # 最后 w 帧
        first_frames = gray_small[0: w]      # 最前 w 帧
        
        # 用 L1 距离计算差异
        diff = (last_frames - first_frames).abs().mean()
        
        # 转换为相似度（灰度范围 [0, 1]）
        similarity = float(torch.clamp(1.0 - diff, 0.0, 1.0))
        
        return similarity
    
    def _motion_autocorr(self, gray_small: torch.Tensor) -> torch.Tensor:
        """
        计算运动的自相关（基于帧差能量）
        
        用于抑制"几乎没动"的微循环，增强有运动的真实循环
        
        Args:
            gray_small: 灰度图像 (N, H, W)
            
        Returns:
            运动自相关序列 (N-1,)
        """
        # 计算一阶帧差的能量序列
        frame_diff = (gray_small[1:] - gray_small[:-1]).abs().mean(dim=(1, 2))  # (N-1,)
        
        # 零均值化
        d = (frame_diff - frame_diff.mean()) / (frame_diff.std() + 1e-6)
        
        M = d.shape[0]
        
        # 使用 FFT 计算自相关
        # 补零到 2M 确保线性相关
        fft_result = torch.fft.rfft(d, n=2 * M)
        power_spectrum = (fft_result * fft_result.conj()).real
        autocorr = torch.fft.irfft(power_spectrum, n=2 * M)[:M]
        
        # 归一化
        autocorr = autocorr / (autocorr[0] + 1e-9)
        autocorr[0] = 0.0  # 忽略零延迟
        
        return autocorr  # (M,)
    
    def _prepare_for_analysis(
        self, 
        frames: torch.Tensor, 
        stride: int, 
        target_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """预处理：采样和缩放"""
        # 先将 frames 移动到目标设备，避免设备不匹配错误
        frames = frames.to(device)
        
        # 采样
        indices = torch.arange(0, frames.shape[0], stride, device=device)
        sampled = frames.index_select(0, indices)
        
        # 缩放 (NHWC -> NCHW -> resize -> NCHW)
        B, H, W, C = sampled.shape
        scale = target_size / max(H, W)
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        
        # 转换为 NCHW
        sampled_nchw = sampled.permute(0, 3, 1, 2)
        
        # 调整大小
        resized = F.interpolate(
            sampled_nchw,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        
        # 转回 NHWC
        return resized.permute(0, 2, 3, 1)
    
    def _extract_features(
        self, 
        frames: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """
        提取混合特征：灰度块 + 颜色直方图 + 边缘特征
        返回 (N, D) 的特征矩阵
        """
        N, H, W, C = frames.shape
        features_list = []
        
        # 转换为 NCHW 便于处理
        frames_nchw = frames.permute(0, 3, 1, 2)
        
        # 1. 灰度特征 (16x16 网格)
        if C >= 3:
            gray = 0.299 * frames_nchw[:, 0] + 0.587 * frames_nchw[:, 1] + 0.114 * frames_nchw[:, 2]
        else:
            gray = frames_nchw[:, 0]
        
        gray = gray.unsqueeze(1)  # (N, 1, H, W)
        gray_pooled = F.adaptive_avg_pool2d(gray, (16, 16))  # (N, 1, 16, 16)
        gray_feat = gray_pooled.reshape(N, -1)  # (N, 256)
        features_list.append(gray_feat)
        
        # 2. 颜色直方图特征 (简化版)
        if C >= 3:
            # 对每个颜色通道计算均值和标准差
            color_mean = frames_nchw.mean(dim=(2, 3))  # (N, C)
            color_std = frames_nchw.std(dim=(2, 3))    # (N, C)
            color_feat = torch.cat([color_mean, color_std], dim=1)  # (N, 2C)
            features_list.append(color_feat)
        
        # 3. 边缘特征 (Sobel)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=frames.dtype, device=device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        
        edge_mean = edges.mean(dim=(1, 2, 3))  # (N,)
        edge_std = edges.std(dim=(1, 2, 3))    # (N,)
        edge_feat = torch.stack([edge_mean, edge_std], dim=1)  # (N, 2)
        features_list.append(edge_feat)
        
        # 合并所有特征
        features = torch.cat(features_list, dim=1)  # (N, D)
        
        # L2 归一化
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def _compute_autocorrelation(self, features: torch.Tensor) -> torch.Tensor:
        """
        使用 FFT 计算自相关
        返回归一化的自相关序列
        """
        N, D = features.shape
        
        # 零均值化
        features_centered = features - features.mean(dim=0, keepdim=True)
        
        # 转置以便对每个维度做 FFT
        features_t = features_centered.T  # (D, N)
        
        # FFT -> 功率谱 -> IFFT
        fft_result = torch.fft.rfft(features_t, n=2*N, dim=1)
        power_spectrum = (fft_result * fft_result.conj()).real
        autocorr_per_dim = torch.fft.irfft(power_spectrum, n=2*N, dim=1)
        
        # 在维度上求和
        autocorr = autocorr_per_dim.sum(dim=0)[:N]  # 只取前 N 个
        
        # 归一化到 [0, 1]
        autocorr = autocorr / (autocorr[0] + 1e-9)
        autocorr[0] = 0.0  # 忽略零延迟
        
        return autocorr
    
    def _detect_period(
        self,
        autocorr: torch.Tensor,
        N: int,
        min_period: int,
        max_period: int,
        motion_ac: Optional[torch.Tensor] = None,
        min_pairs: int = 12,
        prefer_longer: bool = True,
        length_bias: float = 0.05,
        motion_weight: float = 0.25,
    ) -> Tuple[int, float, list]:
        """
        基于自相关 + 样本对数惩罚 + 运动周期性 + 长度偏好的综合评分选周期
        
        Args:
            autocorr: 自相关序列
            N: 原始帧数（用于计算样本对数）
            min_period: 最小周期
            max_period: 最大周期
            motion_ac: 运动自相关序列（可选）
            min_pairs: 最小成对样本数要求
            prefer_longer: 是否优先选择长周期
            length_bias: 长度偏好强度
            motion_weight: 运动周期性权重
            
        Returns:
            (period, confidence, candidates)
        """
        kmin = int(min_period)
        kmax = int(min(max_period, len(autocorr) - 1))
        
        if kmax < kmin:
            return kmin, 0.0, []
        
        # 提取搜索范围
        a = autocorr[kmin:kmax+1]  # (K,)
        
        if len(a) == 0:
            return kmin, 0.0, []
        
        # 平滑处理
        kernel_size = 5
        kernel = torch.ones(kernel_size, device=a.device) / kernel_size
        a_sm = F.conv1d(
            a.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=kernel_size // 2
        ).squeeze()
        
        # 实际的 lag 值
        ks = torch.arange(kmin, kmax+1, device=a.device)
        
        # 样本对数惩罚：N-k 越小，统计越不稳定
        pairs = (N - ks).clamp(min=1)
        pair_fac = torch.sqrt(pairs / pairs.max())  # [0,1]
        
        # 基础分数 = 平滑自相关 × 样本对惩罚
        score = a_sm * pair_fac
        
        # 融合运动周期性（对"细微抖动"周期降权）
        if motion_ac is not None and len(motion_ac) > 1:
            # 提取对应的运动自相关切片
            m_start = max(0, kmin - 1)
            m_end = min(len(motion_ac), kmax)
            m_slice = motion_ac[m_start:m_end] if m_end > m_start else None
            
            if m_slice is not None and m_slice.numel() == score.numel():
                # 归一化到 [0,1]
                m_min = m_slice.min()
                m_max = m_slice.max()
                m_norm = (m_slice - m_min) / (m_max - m_min + 1e-9)
                
                # 混合分数
                score = (1.0 - motion_weight) * score + motion_weight * m_norm
        
        # 轻度长度偏好（避免8帧微循环抢峰）
        if prefer_longer and (kmax - kmin) > 1 and length_bias > 0.0:
            len_norm = (ks - kmin).float() / max(1, (kmax - kmin))
            score = score + length_bias * len_norm
        
        # 计算稳健置信度（使用中位数和 IQR）
        med = score.median()
        q75_idx = int(0.75 * score.numel())
        q25_idx = int(0.25 * score.numel())
        q75 = score.kthvalue(min(q75_idx, score.numel())).values
        q25 = score.kthvalue(max(1, q25_idx)).values
        iqr = q75 - q25 + 1e-9
        
        # 过滤成对样本太少的周期
        valid = (pairs >= min_pairs)
        score = torch.where(valid, score, torch.full_like(score, -1e9))
        
        # 找所有局部峰值
        peaks = []
        s = score
        for i in range(1, s.numel() - 1):
            if s[i] > s[i-1] and s[i] > s[i+1]:
                # 稳健显著性
                prom = float((s[i] - med) / iqr)
                if prom > 0.0:
                    peaks.append({
                        "k": int(ks[i].item()),
                        "score": float(s[i].item()),
                        "prom": float(prom)
                    })
        
        # 如果没有找到峰值，退化到全局最大
        if not peaks:
            i = int(torch.argmax(s))
            k_sel = int(ks[i].item())
            conf = float((s[i] - med) / iqr)
            return k_sel, conf, [{"period": k_sel, "score": float(s[i].item()), "prominence": conf}]
        
        # 最长优先 within Δ（在分数接近的候选中选最长的）
        peaks.sort(key=lambda x: x["score"], reverse=True)
        s_max = peaks[0]["score"]
        margin = 0.05 * max(1.0, abs(s_max))
        
        # 从长到短排序
        peaks_sorted = sorted(peaks, key=lambda x: x["k"], reverse=True)
        
        chosen = None
        for p in peaks_sorted:
            if (s_max - p["score"]) <= margin:
                chosen = p
                break
        
        if chosen is None:
            chosen = peaks[0]
        
        k_sel = chosen["k"]
        conf = chosen["prom"]  # 稳健置信度
        
        # 包装候选列表
        cands = [{"period": p["k"], "score": p["score"], "prominence": p["prom"]} for p in peaks[:5]]
        
        return k_sel, float(conf), cands
    
    def _find_best_start(
        self, 
        frames: torch.Tensor, 
        period: int, 
        device: torch.device,
        window: int = 3
    ) -> int:
        """
        通过比较接缝相似度找到最佳起始点
        """
        N = frames.shape[0]
        
        if period >= N:
            return 0
        
        # 提取简单特征用于快速比较
        frames_device = frames.to(device)
        
        # 转灰度
        if frames.shape[-1] >= 3:
            gray = (0.299 * frames_device[..., 0] + 
                   0.587 * frames_device[..., 1] + 
                   0.114 * frames_device[..., 2])
        else:
            gray = frames_device[..., 0]
        
        # 下采样加速
        gray_small = F.interpolate(
            gray.unsqueeze(1),  # (N, 1, H, W)
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # (N, H, W)
        
        best_start = 0
        best_score = -float('inf')
        
        # 在 [0, period) 范围内搜索
        for start in range(period):
            if start + period >= N:
                break
            
            # 计算接缝窗口的相似度
            score = 0.0
            count = 0
            
            for offset in range(-window, window + 1):
                idx1 = start + offset
                idx2 = start + period + offset
                
                if 0 <= idx1 < N and 0 <= idx2 < N:
                    # L1 距离（越小越好，取负值）
                    diff = (gray_small[idx1] - gray_small[idx2]).abs().mean()
                    score -= float(diff)
                    count += 1
            
            if count > 0:
                score /= count
                
                if score > best_score:
                    best_score = score
                    best_start = start
        
        return best_start
    
    def _refine_boundaries(
        self,
        frames: torch.Tensor,
        start: int,
        period: int,
        device: torch.device,
        radius: int = 4
    ) -> Tuple[int, int]:
        """
        在小范围内微调起点和周期
        """
        N = frames.shape[0]
        
        if start + period >= N or radius == 0:
            return start, period
        
        frames_device = frames.to(device)
        
        # 转灰度并下采样
        if frames.shape[-1] >= 3:
            gray = (0.299 * frames_device[..., 0] + 
                   0.587 * frames_device[..., 1] + 
                   0.114 * frames_device[..., 2])
        else:
            gray = frames_device[..., 0]
        
        gray_small = F.interpolate(
            gray.unsqueeze(1),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        best_start = start
        best_period = period
        best_sim = -float('inf')
        
        # 在小范围内搜索
        for ds in range(-radius, radius + 1):
            for dp in range(-radius, radius + 1):
                new_start = start + ds
                new_period = period + dp
                
                if new_start < 0 or new_start + new_period >= N or new_period < 2:
                    continue
                
                # 计算首尾帧相似度
                first_frame = gray_small[new_start]
                last_frame = gray_small[new_start + new_period - 1]
                
                # SSIM 的简化版：相关系数
                sim = F.cosine_similarity(
                    first_frame.flatten().unsqueeze(0),
                    last_frame.flatten().unsqueeze(0),
                    dim=1
                )
                
                if sim > best_sim:
                    best_sim = sim
                    best_start = new_start
                    best_period = new_period
        
        return best_start, best_period
    
    def _extract_frames(
        self,
        frames: torch.Tensor,
        loop_start: int,
        loop_period: int
    ) -> torch.Tensor:
        """
        提取循环帧
        """
        N = frames.shape[0]
        loop_end = min(loop_start + loop_period, N)
        loop_frames = frames[loop_start:loop_end]
        return loop_frames
    
    def _generate_report(
        self,
        total_frames: int,
        loop_start: int,
        loop_period: int,
        confidence: float,
        candidates: list
    ) -> str:
        """生成分析报告"""
        report = f"""循环检测报告
==================
总帧数: {total_frames}
循环起始: {loop_start}
循环周期: {loop_period} 帧
循环结束: {loop_start + loop_period}
置信度: {confidence:.3f}

候选周期:
"""
        for i, cand in enumerate(candidates[:3], 1):
            report += f"  {i}. 周期={cand['period']}, 得分={cand['score']:.3f}, 显著性={cand['prominence']:.3f}\n"
        
        return report
    
    def _return_all_frames(
        self, 
        frames: torch.Tensor, 
        reason: str
    ) -> Tuple:
        """
        返回全部帧（未检测到循环时的降级策略）
        """
        N = frames.shape[0]
        
        report = f"""循环检测报告
==================
状态: 未检测到循环
原因: {reason}
总帧数: {N}
操作: 返回全部帧
"""
        
        return (
            frames,  # 全部帧作为 loop_frames
            0,       # loop_start
            N,       # loop_period (整个视频)
            0.0,     # confidence
            report
        )


# 节点注册
NODE_CLASS_MAPPINGS = {
    "LoopDetectExtract": LoopDetectExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopDetectExtract": "🔁 循环检测与提取",
}

