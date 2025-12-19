"""
视频帧提取节点 - ComfyUI 节点层
"""

from ..core.frame_extractor import FrameExtractor


class VideoFrameExtractor:
    """
    视频帧提取节点（ComfyUI 接口）

    通过实例化 FrameExtractor 来执行核心逻辑
    """

    DESCRIPTION = "提取视频首尾帧并颠倒顺序，生成 VAE 蒙版和遮罩视频，用于视频循环优化。"

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
                "crossfade_frames": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "首尾各提取用于交叉溶解的帧数"
                    }
                ),
                "mask_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "中间蒙版区的帧数"
                    }
                ),
                "discard_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "首尾各丢弃的帧数（填充灰色让VAE重建）"
                    }
                ),
                "fill_color": (
                    "STRING",
                    {
                        "default": "#7F7F7F",
                        "tooltip": "填充颜色（十六进制，如 #7F7F7F）"
                    }
                ),
                "ensure_4n_plus_1": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "自动调整帧数满足 4n+1 (VAE编码要求)，不足的帧数会增加到中间蒙版区"
                    }
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = (
        "head_frames",
        "tail_frames",
        "middle_frames",
        "mask",
        "masked_video"
    )
    FUNCTION = "extract_frames"
    CATEGORY = "kivi_nodes"

    def __init__(self):
        """初始化时创建一次 FrameExtractor 实例"""
        self._extractor = FrameExtractor()

    def extract_frames(
        self,
        frames,
        crossfade_frames,
        mask_frames,
        discard_frames,
        fill_color,
        ensure_4n_plus_1=True
    ):
        """执行帧提取，调用核心 FrameExtractor 实例"""
        return self._extractor.extract_frames(
            frames=frames,
            crossfade_frames=crossfade_frames,
            mask_frames=mask_frames,
            discard_frames=discard_frames,
            fill_color=fill_color,
            ensure_4n_plus_1=ensure_4n_plus_1
        )


# # 节点注册
# NODE_CLASS_MAPPINGS = {
#     "VideoFrameExtractor": VideoFrameExtractor,
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "VideoFrameExtractor": "📹 视频循环帧提取器",
# }