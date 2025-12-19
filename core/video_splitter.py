"""
视频分割节点 - 核心逻辑
根据索引位置将视频序列分割为两部分
"""

import torch
from typing import Tuple


class VideoSplitter:
    """
    根据索引位置分割视频序列
    """

    DESCRIPTION = "根据指定的索引位置将视频序列分割为两部分。"

    def split_video(
        self,
        frames: torch.Tensor,
        split_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分割视频序列

        Args:
            frames: 输入帧序列，形状 (N, H, W, C)
            split_index: 分割位置索引（从0开始）

        Returns:
            (part1, part2) - 分割后的两部分
            - part1: [0, split_index)
            - part2: [split_index, N)
        """
        N = frames.shape[0]
        device = frames.device
        dtype = frames.dtype

        # 边界检查和调整
        if split_index < 0:
            split_index = 0
        elif split_index > N:
            split_index = N

        # 分割视频
        if split_index == 0:
            # 分割点在开头，第一部分为空
            H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]
            part1 = torch.zeros((1, H, W, C), dtype=dtype, device=device)
            part2 = frames
        elif split_index >= N:
            # 分割点在末尾或超出，第二部分为空
            H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]
            part1 = frames
            part2 = torch.zeros((1, H, W, C), dtype=dtype, device=device)
        else:
            # 正常分割
            part1 = frames[:split_index]
            part2 = frames[split_index:]

        return (part1, part2)
