"""
通用工具函数
"""
import torch
import numpy as np
from typing import Tuple, Optional
from PIL import Image

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将 torch 张量转换为 PIL 图像
    """
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    img_np = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    
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
    """
    if image.mode not in ('RGB', 'RGBA', 'L'):
        image = image.convert('RGB')
    
    img_np = np.array(image).astype(np.float32) / 255.0
    
    if img_np.ndim == 2:
        img_np = img_np[:, :, np.newaxis]
    
    return torch.from_numpy(img_np)

def get_device() -> torch.device:
    """获取合适的计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
