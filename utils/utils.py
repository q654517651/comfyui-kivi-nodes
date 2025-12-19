"""
工具函数模块
包含节点开发中常用的辅助函数
"""

import torch
import numpy as np
from typing import Tuple, Optional
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将 torch 张量转换为 PIL 图像
    
    参数:
        tensor: 形状为 (H, W, C) 的张量，值范围 [0, 1]
    
    返回:
        PIL.Image 对象
    """
    # 确保是 CPU 张量
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # 转换为 numpy
    img_np = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # 转换为 PIL
    if img_np.shape[-1] == 1:
        return Image.fromarray(img_np[:, :, 0], mode='L')
    elif img_np.shape[-1] == 3:
        return Image.fromarray(img_np, mode='RGB')
    elif img_np.shape[-1] == 4:
        return Image.fromarray(img_np, mode='RGBA')
    else:
        raise ValueError(f"不支持的通道数: {img_np.shape[-1]}")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    将 PIL 图像转换为 torch 张量
    
    参数:
        image: PIL.Image 对象
    
    返回:
        形状为 (H, W, C) 的张量，值范围 [0, 1]
    """
    # 转换为 RGB
    if image.mode not in ('RGB', 'RGBA', 'L'):
        image = image.convert('RGB')
    
    # 转换为 numpy
    img_np = np.array(image).astype(np.float32) / 255.0
    
    # 确保是 3 维
    if img_np.ndim == 2:
        img_np = img_np[:, :, np.newaxis]
    
    # 转换为 tensor
    return torch.from_numpy(img_np)


def resize_image_tensor(
    image: torch.Tensor,
    width: int,
    height: int,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    调整图像张量大小
    
    参数:
        image: 形状为 (B, H, W, C) 的图像张量
        width: 目标宽度
        height: 目标高度
        mode: 插值模式 ('nearest', 'bilinear', 'bicubic')
    
    返回:
        调整大小后的张量
    """
    # 转换为 (B, C, H, W)
    img = image.permute(0, 3, 1, 2)
    
    # 调整大小
    import torch.nn.functional as F
    resized = F.interpolate(
        img,
        size=(height, width),
        mode=mode,
        align_corners=False if mode != 'nearest' else None
    )
    
    # 转换回 (B, H, W, C)
    return resized.permute(0, 2, 3, 1)


def normalize_tensor(tensor: torch.Tensor, target_min: float = 0.0, target_max: float = 1.0) -> torch.Tensor:
    """
    归一化张量到指定范围
    
    参数:
        tensor: 输入张量
        target_min: 目标最小值
        target_max: 目标最大值
    
    返回:
        归一化后的张量
    """
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val - min_val < 1e-7:
        return torch.full_like(tensor, (target_min + target_max) / 2)
    
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min


def get_batch_chunks(total: int, chunk_size: int) -> list:
    """
    将总数分割成批次
    
    参数:
        total: 总数量
        chunk_size: 每批大小
    
    返回:
        [(start, end), ...] 批次列表
    """
    if chunk_size <= 0 or chunk_size >= total:
        return [(0, total)]
    
    chunks = []
    for i in range(0, total, chunk_size):
        chunks.append((i, min(i + chunk_size, total)))
    
    return chunks


def blend_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    alpha: float,
    mode: str = 'normal'
) -> torch.Tensor:
    """
    混合两张图像
    
    参数:
        img1: 第一张图像
        img2: 第二张图像
        alpha: 混合比例 [0, 1]
        mode: 混合模式 ('normal', 'multiply', 'screen', 'add')
    
    返回:
        混合后的图像
    """
    alpha = max(0.0, min(1.0, alpha))
    
    if mode == 'normal':
        result = img1 * (1 - alpha) + img2 * alpha
    elif mode == 'multiply':
        result = img1 * (1 - alpha) + (img1 * img2) * alpha
    elif mode == 'screen':
        result = img1 * (1 - alpha) + (1 - (1 - img1) * (1 - img2)) * alpha
    elif mode == 'add':
        result = img1 * (1 - alpha) + torch.clamp(img1 + img2, 0, 1) * alpha
    else:
        raise ValueError(f"未知的混合模式: {mode}")
    
    return torch.clamp(result, 0, 1)


def apply_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    invert: bool = False
) -> torch.Tensor:
    """
    应用遮罩到图像
    
    参数:
        image: 形状为 (B, H, W, C) 的图像
        mask: 形状为 (B, H, W) 的遮罩
        invert: 是否反转遮罩
    
    返回:
        应用遮罩后的图像
    """
    # 确保遮罩形状匹配
    if mask.dim() == 3:
        mask = mask.unsqueeze(-1)  # (B, H, W, 1)
    
    # 反转遮罩
    if invert:
        mask = 1 - mask
    
    # 应用遮罩
    return image * mask


def crop_image(
    image: torch.Tensor,
    x: int,
    y: int,
    width: int,
    height: int
) -> torch.Tensor:
    """
    裁剪图像
    
    参数:
        image: 形状为 (B, H, W, C) 的图像
        x: 起始 x 坐标
        y: 起始 y 坐标
        width: 裁剪宽度
        height: 裁剪高度
    
    返回:
        裁剪后的图像
    """
    _, img_h, img_w, _ = image.shape
    
    # 确保坐标在有效范围内
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    
    # 确保裁剪区域不超出图像
    width = min(width, img_w - x)
    height = min(height, img_h - y)
    
    return image[:, y:y+height, x:x+width, :]


def pad_image(
    image: torch.Tensor,
    top: int,
    right: int,
    bottom: int,
    left: int,
    color: Tuple[float, float, float] = (0, 0, 0)
) -> torch.Tensor:
    """
    填充图像边缘
    
    参数:
        image: 形状为 (B, H, W, C) 的图像
        top, right, bottom, left: 各边填充像素数
        color: 填充颜色 (R, G, B)，值范围 [0, 1]
    
    返回:
        填充后的图像
    """
    B, H, W, C = image.shape
    
    # 创建新画布
    new_h = H + top + bottom
    new_w = W + left + right
    
    # 创建填充颜色张量
    color_tensor = torch.tensor(color[:C], dtype=image.dtype, device=image.device)
    canvas = color_tensor.view(1, 1, 1, C).expand(B, new_h, new_w, C).clone()
    
    # 放置原图
    canvas[:, top:top+H, left:left+W, :] = image
    
    return canvas


def get_device() -> torch.device:
    """
    获取可用的计算设备
    
    返回:
        torch.device 对象
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def free_memory():
    """清理 GPU 内存"""
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def print_tensor_info(tensor: torch.Tensor, name: str = "Tensor"):
    """
    打印张量信息（用于调试）
    
    参数:
        tensor: 要检查的张量
        name: 张量名称
    """
    print(f"\n{'='*50}")
    print(f"{name} 信息:")
    print(f"  形状: {tensor.shape}")
    print(f"  数据类型: {tensor.dtype}")
    print(f"  设备: {tensor.device}")
    print(f"  值范围: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"  平均值: {tensor.mean().item():.4f}")
    print(f"  标准差: {tensor.std().item():.4f}")
    print(f"{'='*50}\n")


class ProgressBar:
    """简单的进度条封装"""
    
    def __init__(self, total: int, desc: str = "处理中"):
        """
        参数:
            total: 总步数
            desc: 描述文字
        """
        try:
            import comfy.utils
            self.pbar = comfy.utils.ProgressBar(total)
            self.comfy_mode = True
        except:
            self.pbar = None
            self.comfy_mode = False
            self.current = 0
            self.total = total
            self.desc = desc
    
    def update(self, n: int = 1):
        """更新进度"""
        if self.comfy_mode:
            self.pbar.update(n)
        else:
            self.current += n
            percent = (self.current / self.total) * 100
            print(f"\r{self.desc}: {self.current}/{self.total} ({percent:.1f}%)", end="")
            if self.current >= self.total:
                print()  # 换行

