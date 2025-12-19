"""
ComfyUI Kivi Nodes - 节点定义层
所有 ComfyUI 节点的注册和导出
"""

from .loop_detector import LoopDetectExtract
from .frame_extractor import VideoFrameExtractor
from .loop_crossfade import LoopVideoCrossfade
from .video_extractors import VideoConcatExtractor
from .video_splitter import VideoSplit

# 节点类映射 - ComfyUI 使用这个字典来识别节点
NODE_CLASS_MAPPINGS = {
    "LoopDetectExtract": LoopDetectExtract,
    "VideoFrameExtractor": VideoFrameExtractor,
    "LoopVideoCrossfade": LoopVideoCrossfade,
    "VideoConcatExtractor": VideoConcatExtractor,
    "VideoSplit": VideoSplit,
}

# 节点显示名称映射 - 在 ComfyUI 界面中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopDetectExtract": "循环检测与提取",
    "VideoFrameExtractor": "视频帧提取器",
    "LoopVideoCrossfade": "循环视频交叉溶解",
    "VideoConcatExtractor": "视频拼接帧提取器",
    "VideoSplit": "视频分割",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

