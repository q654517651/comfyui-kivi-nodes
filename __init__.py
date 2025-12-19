# ComfyUI 自定义节点 - 主入口
# 从 nodes 包导入所有节点定义和映射

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 必须导出这两个映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
