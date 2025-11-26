# ComfyUI Kivi Nodes

> **注意**: 请在后续的所有交互和回答中默认使用**中文**。

## 项目概述
**ComfyUI Kivi Nodes** 是一个为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 设计的自定义节点集合，主要专注于视频分析和处理。其核心功能是 **循环检测与提取 (Loop Detect & Extract)** 系统，该系统利用 GPU 加速的 FFT（快速傅里叶变换）技术，智能识别并提取视频序列中的无缝循环片段。

## 核心文件与节点

### 主要功能
*   **`loop_detector.py` (节点: `LoopDetectExtract`)**:
    *   **用途**: 分析视频帧序列以寻找最佳循环点。
    *   **特性**: GPU 加速、自动相位锁定、置信度评分以及智能回退策略。
    *   **输出**: 循环帧、前/后上下文帧以及分析元数据。
*   **`loop_crossfade.py` (节点: `LoopVideoCrossfade`)**:
    *   **用途**: 使用交叉溶解技术在视频循环之间创建平滑过渡。
*   **`frame_extractor.py` (节点: `VideoFrameExtractor`)**:
    *   **用途**: 用于从视频源中提取特定帧进行处理的工具。

### 支持与示例
*   **`__init__.py`**: ComfyUI 的入口点。注册所有节点类和显示名称。
*   **`example_nodes.py`**: 包含用于演示的 `ExampleImageNode` 和 `ExampleTextNode`。
*   **`utils.py`**: 共享工具函数（主要是张量/图像操作）。
*   **`comfyui-WhiteRabbit-master/`**: 一个包含示例代码的参考目录（不属于活动节点套件的一部分）。

## 安装与使用

### 安装
1.  将此仓库克隆到您的 `ComfyUI/custom_nodes/` 目录中。
2.  安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```
    *(注意: `torch`, `numpy`, 和 `pillow` 通常已预装在 ComfyUI 环境中)*。

### 依赖项
*   **Python 3.x**
*   **Torch**: 用于张量运算和 GPU 加速。
*   **NumPy & Pillow**: 用于图像处理和数值分析。

## 开发指南

### 添加新节点
1.  创建一个新的 `.py` 文件（或添加到现有文件中）。
2.  定义类，包含 `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, 和 `CATEGORY`。
3.  在 `__init__.py` 的 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 中注册节点。

### 节点映射公约
*   **类映射 (Class Mapping)**: 将内部类名映射到 Python 类。
*   **显示名称 (Display Name)**: 将内部类名映射到 UI 可见的名称（支持 Emoji 和非英文字符）。

### 目录结构说明
`comfyui-WhiteRabbit-master/` 目录仅作为参考实现包含在内，除非专门移植功能，否则在开发核心 `Kivi` 节点时应忽略该目录。