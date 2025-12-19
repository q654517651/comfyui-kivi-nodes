"""
循环视频交叉溶解节点 - ComfyUI 节点层
"""

from ..core.loop_crossfade import VideoCrossfader


class LoopVideoCrossfade:
    """
    循环视频交叉溶解节点（ComfyUI 接口）

    通过实例化 VideoCrossfader 来执行核心逻辑
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

    def __init__(self):
        """初始化时创建一次 VideoCrossfader 实例"""
        self._crossfader = VideoCrossfader()

    def crossfade_merge(
        self,
        video1,
        video2,
        crossfade_frames,
        enable_crossfade=True
    ):
        """执行交叉溶解，调用核心 VideoCrossfader 实例"""
        return self._crossfader.crossfade_merge(
            video1=video1,
            video2=video2,
            crossfade_frames=crossfade_frames,
            enable_crossfade=enable_crossfade
        )


# # 节点注册
# NODE_CLASS_MAPPINGS = {
#     "LoopVideoCrossfade": LoopVideoCrossfade,
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "LoopVideoCrossfade": "🔄 循环视频交叉溶解",
# }