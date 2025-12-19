"""
视频拼接帧提取节点 - 核心逻辑
提取视频序列的前 N 帧和后 N 帧，保持原顺序（用于拼接场景）
"""

import torch
from typing import Tuple


class ConcatFrameExtractor:
    """
    提取视频序列的首尾帧，保持原顺序，生成用于 VAE 的蒙版和遮罩视频

    与 FrameExtractor 的区别（完全相反）：
    - FrameExtractor（循环场景）: 丢弃首尾"外侧"帧（a和e），保留"内侧"帧（b和d）
    - ConcatFrameExtractor（拼接场景）: 保留首尾"外侧"帧（a和d），丢弃"内侧"帧（b和c）

    使用场景：
    - video1 经过此提取器，尾部内侧被标记为重建
    - video2 经过此提取器，头部内侧被标记为重建
    - VAE 重建后，video1 的尾部和 video2 的头部可以无缝拼接
    """

    DESCRIPTION = "提取视频首尾帧保持原顺序，生成 VAE 蒙版和遮罩视频，用于视频拼接场景。"

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
        提取首尾帧并保持原顺序，生成 VAE 蒙版

        输出顺序：a + b + mask + c + d

        拼接场景的分段逻辑（与循环场景相反）：
        - a（头部外侧）：保留原视频
        - b（头部内侧）：灰色填充，VAE 重建
        - mask（中间区）：灰色填充，VAE 重建
        - c（尾部内侧）：灰色填充，VAE 重建
        - d（尾部外侧）：保留原视频

        Args:
            frames: 输入帧序列，形状 (N, H, W, C)
            crossfade_frames: 首尾各提取用于交叉溶解的帧数
            mask_frames: 中间蒙版区的帧数
            discard_frames: 内侧丢弃的帧数（b和c的帧数）
            fill_color: 填充颜色（十六进制）
            ensure_4n_plus_1: 是否强制输出帧数为 4n+1

        Returns:
            (head_frames, tail_frames, middle_frames, mask, masked_video)
            注意：为保持接口一致性，head_frames=a, tail_frames=d
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

        # 原视频分段逻辑（拼接场景 - 与循环场景相反）：
        # 保留外侧，丢弃内侧（内侧 = 靠近视频中间的部分）
        #
        # 头部 crossfade 段 [0, crossfade_frames):
        #   a: [0, crossfade_frames - discard_frames) - 头部外侧，保留（原视频）
        #   b: [crossfade_frames - discard_frames, crossfade_frames) - 头部内侧，丢弃（填充纯色）
        #
        # 尾部 crossfade 段 [N-crossfade_frames, N):
        #   c: [N-crossfade_frames, N-crossfade_frames+discard_frames) - 尾部内侧，丢弃（填充纯色）
        #   d: [N-crossfade_frames+discard_frames, N) - 尾部外侧，保留（原视频）

        # 提取 a (head_frames 外侧) - 头部外侧的原视频
        a_start = 0
        a_end = crossfade_frames - discard_frames
        if a_end > 0 and a_end <= N:
            a = frames[a_start:a_end]
            a_count = a_end - a_start
        else:
            a = placeholder
            a_count = 0

        # b 是头部内侧，丢弃（纯色填充，不从原视频提取）
        b_count = discard_frames

        # 提取 d (tail_frames 外侧) - 尾部外侧的原视频
        d_start = N - crossfade_frames + discard_frames
        d_end = N
        if d_start >= 0 and d_start < d_end and d_end <= N:
            d = frames[d_start:d_end]
            d_count = d_end - d_start
        else:
            d = placeholder
            d_count = 0

        # c 是尾部内侧，丢弃（纯色填充，不从原视频提取）
        c_count = discard_frames

        # 提取中间部分（原视频中间部分，作为 middle_frames 返回，供 debug 使用）
        middle_start = crossfade_frames
        middle_end = N - crossfade_frames
        if middle_end > middle_start:
            middle = frames[middle_start:middle_end]
        else:
            middle = placeholder

        # 计算基础输出长度
        output_length = a_count + b_count + mask_frames + c_count + d_count

        # 自动调整帧数为 4n+1
        if ensure_4n_plus_1 and output_length > 0:
            remainder = (output_length - 1) % 4
            if remainder != 0:
                needed = 4 - remainder
                mask_frames += needed
                output_length += needed

        # 生成 mask 和 masked_video
        # 输出顺序: a + b + mask + c + d
        # 蒙版: [黑a | 白b | 白mask | 白c | 黑d]
        # 遮罩: [a原视频 | 灰b | 灰mask | 灰c | d原视频]
        # 说明: 保留外侧（a和d），丢弃内侧（b和c）让VAE重建以便拼接

        if output_length == 0:
            # 没有输出，返回占位符
            mask = torch.zeros((1, H, W), dtype=dtype, device=device)
            masked_video = torch.zeros((1, H, W, C), dtype=dtype, device=device)
            return (placeholder, placeholder, placeholder, mask, masked_video)

        # 初始化 mask 和 masked_video
        mask = torch.zeros((output_length, H, W), dtype=dtype, device=device)
        masked_video = torch.zeros((output_length, H, W, C), dtype=dtype, device=device)

        idx = 0

        # 1. a (头部外侧) - 黑色蒙版，原视频
        if a_count > 0:
            mask[idx:idx+a_count] = 0.0  # 黑色（保留）
            masked_video[idx:idx+a_count] = a
            idx += a_count

        # 2. b (头部内侧) - 白色蒙版，灰色视频
        if b_count > 0:
            mask[idx:idx+b_count] = 1.0  # 白色（重建）
            masked_video[idx:idx+b_count] = gray_color.view(1, 1, 1, 3)
            idx += b_count

        # 3. mask区 - 白色蒙版，灰色视频
        if mask_frames > 0:
            mask[idx:idx+mask_frames] = 1.0  # 白色（重建）
            masked_video[idx:idx+mask_frames] = gray_color.view(1, 1, 1, 3)
            idx += mask_frames

        # 4. c (尾部内侧) - 白色蒙版，灰色视频
        if c_count > 0:
            mask[idx:idx+c_count] = 1.0  # 白色（重建）
            masked_video[idx:idx+c_count] = gray_color.view(1, 1, 1, 3)
            idx += c_count

        # 5. d (尾部外侧) - 黑色蒙版，原视频
        if d_count > 0:
            mask[idx:idx+d_count] = 0.0  # 黑色（保留）
            masked_video[idx:idx+d_count] = d
            idx += d_count

        # 返回：head_frames(a), tail_frames(d), middle_frames(中间部分), mask, masked_video
        return (a, d, middle, mask, masked_video)
