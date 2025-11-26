# ComfyUI Kivi Nodes

**ComfyUI Kivi Nodes** 是一套专注于视频分析与处理的自定义节点，旨在为 ComfyUI 用户提供强大的循环视频制作与优化工具。

## ✨ 核心功能

### 1. 🔁 循环检测与提取 (Loop Detect & Extract)
利用 GPU 加速的 FFT 算法，智能分析视频帧序列，自动寻找并提取最佳的无缝循环片段。
- **主要特性**:
    - 自动相位锁定与置信度评分
    - 智能回退策略（未检测到循环时自动调整）
    - 详细的分析报告输出

### 2. 🔄 循环视频交叉溶解 (Loop Video Crossfade)
专为循环视频设计的交叉溶解工具，修复了传统溶解导致的循环断点问题。
- **主要特性**:
    - 正确的循环时间线溶解逻辑 (`f -> a` 渐变)
    - 支持自定义溶解帧数与曲线

### 3. 📹 视频帧提取器 (Video Frame Extractor)
辅助工具，用于从长视频中精确提取片段，并支持自动补帧以满足 VAE 编码需求（4n+1 规则）。

## 📦 安装

1. 进入你的 ComfyUI `custom_nodes` 目录：
    ```bash
    cd ComfyUI/custom_nodes
    ```
2. 克隆本项目：
    ```bash
    git clone https://github.com/yourusername/ComfyUI_Kivi_Nodes.git
    ```
3. 重启 ComfyUI。

*注意：本项目仅依赖 ComfyUI 原生环境 (torch, numpy, pillow)，无需额外安装复杂依赖。*

## 🛠️ 节点列表

| 节点名称 | 英文 ID | 分类 | 描述 |
| :--- | :--- | :--- | :--- |
| **循环检测与提取** | `LoopDetectExtract` | `kivi_nodes` | 自动检测并提取视频循环片段 |
| **循环视频交叉溶解** | `LoopVideoCrossfade` | `kivi_nodes` | 创建无缝的循环视频过渡 |
| **视频帧提取器** | `VideoFrameExtractor` | `kivi_nodes` | 提取帧并处理 VAE 维度要求 |

## 🤝 贡献与开发

欢迎提交 Issue 或 Pull Request。
- 测试脚本位于 `tests/` 目录。
- 项目配置见 `pyproject.toml`。

## 📄 许可证

MIT License