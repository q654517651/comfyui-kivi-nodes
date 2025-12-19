"""
视频分割节点 - ComfyUI 节点层
"""

from ..core.video_splitter import VideoSplitter


class VideoSplit:
    """
    视频分割节点（ComfyUI 接口）

    通过实例化 VideoSplitter 来执行核心逻辑
    """

    DESCRIPTION = "根据指定的索引位置将视频序列分割为两部分。"

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
                "split_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "分割位置索引（从0开始，分割后第二部分从此帧开始）"
                    }
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("part1", "part2")
    FUNCTION = "split_video"
    CATEGORY = "kivi_nodes"

    def __init__(self):
        """初始化时创建一次 VideoSplitter 实例"""
        self._splitter = VideoSplitter()

    def split_video(self, frames, split_index):
        """执行视频分割，调用核心 VideoSplitter 实例"""
        return self._splitter.split_video(
            frames=frames,
            split_index=split_index
        )
