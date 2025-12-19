"""
视频帧提取节点
提取视频序列的前 N 帧和后 N 帧，并生成 VAE 蒙版
"""

import torch
from typing import Tuple


class FrameExtractor:
    """
    提取视频序列的首尾帧，生成用于 VAE 的蒙版和遮罩视频
    """
    
    DESCRIPTION = "提取视频首尾帧并颠倒顺序，生成 VAE 蒙版和遮罩视频，用于视频循环优化。"
    
    # @classmethod
    # def INPUT_TYPES(cls):
    #     return {
    #         "required": {
    #             "frames": (
    #                 "IMAGE",
    #                 {
    #                     "tooltip": "输入图像序列 (batch, height, width, channels)，值范围 [0, 1]"
    #                 }
    #             ),
    #             "crossfade_frames": (
    #                 "INT",
    #                 {
    #                     "default": 10,
    #                     "min": 0,
    #                     "max": 1000,
    #                     "step": 1,
    #                     "tooltip": "首尾各提取用于交叉溶解的帧数"
    #                 }
    #             ),
    #             "mask_frames": (
    #                 "INT",
    #                 {
    #                     "default": 0,
    #                     "min": 0,
    #                     "max": 1000,
    #                     "step": 1,
    #                     "tooltip": "中间蒙版区的帧数"
    #                 }
    #             ),
    #             "discard_frames": (
    #                 "INT",
    #                 {
    #                     "default": 0,
    #                     "min": 0,
    #                     "max": 1000,
    #                     "step": 1,
    #                     "tooltip": "首尾各丢弃的帧数（填充灰色让VAE重建）"
    #                 }
    #             ),
    #             "fill_color": (
    #                 "STRING",
    #                 {
    #                     "default": "#7F7F7F",
    #                     "tooltip": "填充颜色（十六进制，如 #7F7F7F）"
    #                 }
    #             ),
    #             "ensure_4n_plus_1": (
    #                 "BOOLEAN",
    #                 {
    #                     "default": True,
    #                     "tooltip": "自动调整帧数满足 4n+1 (VAE编码要求)，不足的帧数会增加到中间蒙版区"
    #                 }
    #             ),
    #         }
    #     }
    #
    # RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "IMAGE")
    # RETURN_NAMES = (
    #     "head_frames",
    #     "tail_frames",
    #     "middle_frames",
    #     "mask",
    #     "masked_video"
    # )
    # FUNCTION = "extract_frames"
    # CATEGORY = "kivi_nodes"
    
    def extract_frames(
        self,
        frames: torch.Tensor,
        crossfade_frames: int,
        mask_frames: int,
        discard_frames: int,
        fill_color: str,
        ensure_4n_plus_1: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取首尾帧并颠倒顺序，生成 VAE 蒙版
        
        输出顺序：d + e + mask + a + b
        
        Args:
            frames: 输入帧序列，形状 (N, H, W, C)
            crossfade_frames: 首尾各提取用于交叉溶解的帧数
            mask_frames: 中间蒙版区的帧数
            discard_frames: 首尾各丢弃的帧数
            fill_color: 填充颜色（十六进制）
            ensure_4n_plus_1: 是否强制输出帧数为 4n+1
            
        Returns:
            (head_frames, tail_frames, middle_frames, mask, masked_video)
        """
        N = frames.shape[0]
        H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]
        device = frames.device
        dtype = frames.dtype
        
        # 解析颜色
        fill_color = fill_color.strip()
        if fill_color.startswith('#'):
            fill_color = fill_color[1:]
        try:
            r = int(fill_color[0:2], 16) / 255.0
            g = int(fill_color[2:4], 16) / 255.0
            b = int(fill_color[4:6], 16) / 255.0
            gray_color = torch.tensor([r, g, b], dtype=dtype, device=device)
        except:
            # 默认中灰色
            gray_color = torch.tensor([0.5, 0.5, 0.5], dtype=dtype, device=device)
        
        # 创建占位符（1x1 黑色图片）
        placeholder = torch.zeros((1, 1, 1, C), dtype=dtype, device=device)
        
        # 原视频分段逻辑：
        # crossfade_frames 是首尾各提取的帧数
        # discard_frames 是从 crossfade 段的两端"吃掉"的帧数
        #
        # 头部 crossfade 段 [0, crossfade_frames):
        #   a: [0, discard_frames) - 头部丢弃（填充纯色）
        #   b: [discard_frames, crossfade_frames) - 头部实际使用（原视频）
        #
        # 尾部 crossfade 段 [N-crossfade_frames, N):
        #   d: [N-crossfade_frames, N-discard_frames) - 尾部实际使用（原视频）
        #   e: [N-discard_frames, N) - 尾部丢弃（填充纯色）
        #
        # c: 中间部分（不使用）
        
        # 提取 b (head_frames) - 头部实际使用的原视频
        b_start = discard_frames
        b_end = crossfade_frames
        if b_end <= N and b_start < b_end and discard_frames < crossfade_frames:
            b = frames[b_start:b_end]
            b_count = b_end - b_start
        else:
            b = placeholder
            b_count = 0
        
        # 提取 d (tail_frames) - 尾部实际使用的原视频
        d_start = N - crossfade_frames
        d_end = N - discard_frames
        if d_start >= 0 and d_start < d_end and d_end <= N and discard_frames < crossfade_frames:
            d = frames[d_start:d_end]
            d_count = d_end - d_start
        else:
            d = placeholder
            d_count = 0
        
        # 提取 c (middle_frames)
        c_start = crossfade_frames
        c_end = N - crossfade_frames
        if c_end > c_start:
            c = frames[c_start:c_end]
        else:
            c = placeholder
        
        # a 和 e 的帧数（纯色填充，不从原视频提取）
        a_count = discard_frames
        e_count = discard_frames
        
        # 计算基础输出长度
        output_length = d_count + e_count + mask_frames + a_count + b_count
        
        # 自动调整帧数为 4n+1
        if ensure_4n_plus_1 and output_length > 0:
            # 目标帧数：((output_length - 1) // 4 + 1) * 4 + 1 (如果不满足)
            # 简单算法：
            # target = ceil((x-1)/4)*4 + 1
            # 但是 python 的 // 是向下取整，所以我们要用 math.ceil 或者手动逻辑
            # (output_length - 1) % 4
            
            remainder = (output_length - 1) % 4
            if remainder != 0:
                needed = 4 - remainder
                mask_frames += needed
                output_length += needed
                # print(f"自动调整帧数: 增加 {needed} 帧 mask，总帧数 {output_length - needed} -> {output_length} (满足 4n+1)")
        
        # 生成 mask 和 masked_video
        # 输出顺序: d + e + mask + a + b
        # 蒙版: [黑d | 白e | 白mask | 白a | 黑b]
        # 遮罩: [d原视频 | 灰e | 灰mask | 灰a | b原视频]
        
        if output_length == 0:
            # 没有输出，返回占位符
            mask = torch.zeros((1, H, W), dtype=dtype, device=device)
            masked_video = torch.zeros((1, H, W, C), dtype=dtype, device=device)
            return (placeholder, placeholder, placeholder, mask, masked_video)
        
        # 初始化 mask 和 masked_video
        mask = torch.zeros((output_length, H, W), dtype=dtype, device=device)
        masked_video = torch.zeros((output_length, H, W, C), dtype=dtype, device=device)
        
        idx = 0
        
        # 1. d (尾部实际使用) - 黑色蒙版，原视频
        if d_count > 0:
            mask[idx:idx+d_count] = 0.0  # 黑色（保留）
            masked_video[idx:idx+d_count] = d
            idx += d_count
        
        # 2. e (尾部丢弃段) - 白色蒙版，灰色视频
        if e_count > 0:
            mask[idx:idx+e_count] = 1.0  # 白色（重建）
            masked_video[idx:idx+e_count] = gray_color.view(1, 1, 1, 3)
            idx += e_count
        
        # 3. mask区 - 白色蒙版，灰色视频
        if mask_frames > 0:
            mask[idx:idx+mask_frames] = 1.0  # 白色（重建）
            masked_video[idx:idx+mask_frames] = gray_color.view(1, 1, 1, 3)
            idx += mask_frames
        
        # 4. a (头部丢弃段) - 白色蒙版，灰色视频
        if a_count > 0:
            mask[idx:idx+a_count] = 1.0  # 白色（重建）
            masked_video[idx:idx+a_count] = gray_color.view(1, 1, 1, 3)
            idx += a_count
        
        # 5. b (头部实际使用) - 黑色蒙版，原视频
        if b_count > 0:
            mask[idx:idx+b_count] = 0.0  # 黑色（保留）
            masked_video[idx:idx+b_count] = b
            idx += b_count
        
        return (b, d, c, mask, masked_video)


# # 节点注册
# NODE_CLASS_MAPPINGS = {
#     "VideoFrameExtractor": VideoFrameExtractor,
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "VideoFrameExtractor": "📹 视频帧提取器",
# }

