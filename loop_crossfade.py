"""
循环视频交叉溶解节点
用于无缝拼接两个视频片段
"""

import torch
from typing import Tuple


class LoopVideoCrossfade:
    """
    循环视频交叉溶解
    
    将两个视频序列通过交叉溶解无缝拼接，使输出本身也能循环播放
    """
    
    DESCRIPTION = "通过交叉溶解无缝拼接两个视频序列，输出可循环播放的视频。"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": (
                    "IMAGE",
                    {
                        "tooltip": "第一个视频序列"
                    }
                ),
                "video2": (
                    "IMAGE",
                    {
                        "tooltip": "第二个视频序列"
                    }
                ),
                "crossfade_frames": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "交叉溶解的帧数"
                    }
                ),
                "enable_crossfade": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否启用交叉溶解。关闭时直接拼接，移除首尾重叠帧"
                    }
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("merged_video", "report")
    FUNCTION = "crossfade_merge"
    CATEGORY = "kivi_nodes"
    
    def crossfade_merge(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        crossfade_frames: int,
        enable_crossfade: bool = True
    ) -> Tuple[torch.Tensor, str]:
        """
        交叉溶解合并两个视频
        
        Args:
            video1: 第一个视频序列 [abc]
            video2: 第二个视频序列 [def]
            crossfade_frames: 交叉溶解帧数
            
        Returns:
            (merged_video, report)
        """
        N1 = video1.shape[0]
        N2 = video2.shape[0]
        
        # 检查帧数是否足够
        min_frames = crossfade_frames * 2
        if N1 < min_frames or N2 < min_frames:
            error_msg = f"错误: 视频帧数不足。需要至少 {min_frames} 帧，video1: {N1} 帧，video2: {N2} 帧"
            # 返回 video1 作为降级输出
            return (video1, error_msg)
        
        # 检查尺寸是否匹配
        if video1.shape[1:] != video2.shape[1:]:
            error_msg = f"错误: 视频尺寸不匹配。video1: {video1.shape}, video2: {video2.shape}"
            return (video1, error_msg)
        
        # 分解视频1 [abc]
        a = video1[:crossfade_frames]              # 前 N 帧
        b = video1[crossfade_frames:-crossfade_frames]  # 中间部分
        c = video1[-crossfade_frames:]             # 后 N 帧
        
        # 分解视频2 [def]
        d = video2[:crossfade_frames]              # 前 N 帧
        e = video2[crossfade_frames:-crossfade_frames]  # 中间部分
        f = video2[-crossfade_frames:]             # 后 N 帧
        
        # 根据开关决定是否进行交叉溶解
        if enable_crossfade:
            # 生成 alpha 渐变 (0 -> 1)
            alpha = torch.linspace(0, 1, crossfade_frames, device=video1.device)
            # 调整形状以匹配帧维度 [N, 1, 1, 1]
            alpha = alpha.view(-1, 1, 1, 1)
            
            # 交叉溶解: g = f + a (从 f 过渡到 a)
            # f 权重从 1 -> 0，a 权重从 0 -> 1
            g = f * (1 - alpha) + a * alpha
            
            # 交叉溶解: h = c + d
            # c 权重从 1 -> 0，d 权重从 0 -> 1
            h = c * (1 - alpha) + d * alpha
            
            # 拼接: g + b + h + e
            merged = torch.cat([g, b, h, e], dim=0)
            mode = "交叉溶解模式"
        else:
            # 直接拼接，移除 a 和 c
            # 输出: f + b + d + e
            merged = torch.cat([f, b, d, e], dim=0)
            mode = "直接拼接模式（移除首尾重叠帧）"
        
        # 生成报告
        if enable_crossfade:
            report = f"""循环视频交叉溶解报告
==================
模式: {mode}

video1: {N1} 帧
  - a (开头): {a.shape[0]} 帧
  - b (中间): {b.shape[0]} 帧
  - c (结尾): {c.shape[0]} 帧

video2: {N2} 帧
  - d (开头): {d.shape[0]} 帧
  - e (中间): {e.shape[0]} 帧
  - f (结尾): {f.shape[0]} 帧

交叉溶解: {crossfade_frames} 帧
  - g = crossfade(f, a): {g.shape[0]} 帧
  - h = crossfade(c, d): {h.shape[0]} 帧

输出顺序: g + b + h + e
输出总帧数: {merged.shape[0]} 帧
"""
        else:
            report = f"""循环视频拼接报告
==================
模式: {mode}

video1: {N1} 帧
  - a (开头，已移除): {a.shape[0]} 帧
  - b (中间): {b.shape[0]} 帧
  - c (结尾，已移除): {c.shape[0]} 帧

video2: {N2} 帧
  - d (开头): {d.shape[0]} 帧
  - e (中间): {e.shape[0]} 帧
  - f (结尾): {f.shape[0]} 帧

输出顺序: f + b + d + e
输出总帧数: {merged.shape[0]} 帧

说明: a 和 c 已移除，直接拼接 f、b、d、e
"""
        
        return (merged, report)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "LoopVideoCrossfade": LoopVideoCrossfade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopVideoCrossfade": "🔄 循环视频交叉溶解",
}

