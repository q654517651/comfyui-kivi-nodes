# ComfyUI 自定义节点
# 在这里导入你的节点类

from .loop_detector import (
    LoopDetectExtract,
)
from .frame_extractor import (
    VideoFrameExtractor,
)
from .loop_crossfade import (
    LoopVideoCrossfade,
)

# 节点类映射 - ComfyUI 使用这个字典来识别节点
NODE_CLASS_MAPPINGS = {
    "LoopDetectExtract": LoopDetectExtract,
    "VideoFrameExtractor": VideoFrameExtractor,
    "LoopVideoCrossfade": LoopVideoCrossfade,
}

# 节点显示名称映射 - 在 ComfyUI 界面中显示的名称
# 注意：ComfyUI 的 NODE_DISPLAY_NAME_MAPPINGS 只支持字符串，不支持字典
# 多语言支持需要通过 INPUT_TYPES 中的 tooltip 实现
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopDetectExtract": "🔁 循环检测与提取",
    "VideoFrameExtractor": "📹 视频帧提取器",
    "LoopVideoCrossfade": "🔄 循环视频交叉溶解",
}

# 必须导出这两个映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]