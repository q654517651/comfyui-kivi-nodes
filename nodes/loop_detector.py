"""
循环检测与提取节点 - ComfyUI 节点层
"""

from ..core.loop_detector import LoopDetector


class LoopDetectExtract:
    """
    循环检测与提取节点（ComfyUI 接口）

    通过实例化 LoopDetector 来执行核心逻辑，避免继承导致的频繁实例化问题
    """

    DESCRIPTION = "自动检测视频帧序列中的循环模式，提取循环片段。使用 GPU 加速的 FFT 自相关分析和相似度匹配。"

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
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "循环检测的置信度阈值，低于此值返回全部帧"
                    }
                ),
            },
            "optional": {
                "min_period": (
                    "INT",
                    {
                        "default": 24,
                        "min": 2,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "最小循环周期（帧数）"
                    }
                ),
                "max_period": (
                    "INT",
                    {
                        "default": 300,
                        "min": 4,
                        "max": 5000,
                        "step": 1,
                        "tooltip": "最大循环周期（帧数）"
                    }
                ),
                "analysis_stride": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "分析时的帧采样步长，越大越快但精度降低"
                    }
                ),
                "analysis_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 512,
                        "step": 32,
                        "tooltip": "分析时图像缩放的目标尺寸"
                    }
                ),
                "seam_threshold": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "首尾接缝相似度阈值，超过则优先判定整段闭环"
                    }
                ),
                "min_pairs": (
                    "INT",
                    {
                        "default": 12,
                        "min": 4,
                        "max": 200,
                        "step": 1,
                        "tooltip": "候选周期需满足的最小成对样本数"
                    }
                ),
                "prefer_longer_cycles": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "在分数接近时倾向选择更长周期"
                    }
                ),
                "length_bias": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.01,
                        "tooltip": "对较长周期的轻度偏好强度"
                    }
                ),
                "motion_weight": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "将运动周期性引入候选评分的权重"
                    }
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "loop_frames",
        "loop_start",
        "loop_period",
        "confidence",
        "report"
    )
    FUNCTION = "detect_and_extract"
    CATEGORY = "kivi_nodes"

    def __init__(self):
        """初始化时创建一次 LoopDetector 实例，避免每次调用都重新实例化"""
        self._detector = LoopDetector()

    def detect_and_extract(
        self,
        frames,
        confidence_threshold,
        min_period=24,
        max_period=300,
        analysis_stride=2,
        analysis_size=256,
        seam_threshold=0.85,
        min_pairs=12,
        prefer_longer_cycles=True,
        length_bias=0.05,
        motion_weight=0.25,
    ):
        """执行循环检测和提取，调用核心 LoopDetector 实例"""
        return self._detector.detect_and_extract(
            frames=frames,
            confidence_threshold=confidence_threshold,
            min_period=min_period,
            max_period=max_period,
            analysis_stride=analysis_stride,
            analysis_size=analysis_size,
            seam_threshold=seam_threshold,
            min_pairs=min_pairs,
            prefer_longer_cycles=prefer_longer_cycles,
            length_bias=length_bias,
            motion_weight=motion_weight,
        )



# 旧的节点注册方式（已迁移到 nodes/__init__.py，保留此处仅作参考）
#
# 节点注册
# NODE_CLASS_MAPPINGS = {
#     "LoopDetectExtract": LoopDetectExtract,
#}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "LoopDetectExtract": "🔁 循环检测与提取",
#}