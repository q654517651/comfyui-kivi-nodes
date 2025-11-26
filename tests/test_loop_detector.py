"""
循环检测节点测试脚本
用于验证节点功能是否正常
"""

import torch
import numpy as np


def create_synthetic_loop_video(loop_length=24, num_loops=3, noise_level=0.05):
    """
    创建一个合成的循环视频用于测试
    
    Args:
        loop_length: 单个循环的帧数
        num_loops: 循环重复次数
        noise_level: 噪声水平
    
    Returns:
        torch.Tensor: 形状为 (N, H, W, C) 的图像序列
    """
    H, W, C = 64, 64, 3
    total_frames = loop_length * num_loops
    
    # 创建基础循环模式：渐变色彩变化
    frames = []
    for i in range(loop_length):
        # 创建渐变图案
        t = i / loop_length
        
        # RGB 通道随时间周期性变化
        r = 0.5 + 0.5 * np.sin(2 * np.pi * t)
        g = 0.5 + 0.5 * np.sin(2 * np.pi * t + 2*np.pi/3)
        b = 0.5 + 0.5 * np.sin(2 * np.pi * t + 4*np.pi/3)
        
        # 添加空间渐变
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)
        
        frame = np.stack([
            r * (1 - xx * 0.3),
            g * (1 - yy * 0.3),
            b * np.ones_like(xx)
        ], axis=-1)
        
        frames.append(frame)
    
    # 重复循环
    all_frames = []
    for _ in range(num_loops):
        for frame in frames:
            # 添加少量噪声使其更真实
            noisy_frame = frame + np.random.randn(H, W, C) * noise_level
            noisy_frame = np.clip(noisy_frame, 0, 1)
            all_frames.append(noisy_frame)
    
    # 转换为 torch tensor
    video = torch.from_numpy(np.stack(all_frames, axis=0)).float()
    
    return video, loop_length


def test_loop_detector():
    """测试循环检测节点"""
    print("=" * 60)
    print("循环检测节点测试")
    print("=" * 60)
    
    # 导入节点
    try:
        from loop_detector import LoopDetectExtract
        print("✓ 成功导入 LoopDetectExtract 节点")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return
    
    # 创建测试数据
    print("\n创建测试数据...")
    video, true_loop_length = create_synthetic_loop_video(
        loop_length=20, 
        num_loops=3, 
        noise_level=0.02
    )
    print(f"  视频形状: {video.shape}")
    print(f"  真实循环周期: {true_loop_length} 帧")
    print(f"  总帧数: {video.shape[0]} 帧")
    
    # 初始化节点
    print("\n初始化节点...")
    node = LoopDetectExtract()
    print("✓ 节点初始化成功")
    
    # 运行检测
    print("\n运行循环检测...")
    try:
        results = node.detect_and_extract(
            frames=video,
            context_before=3,
            context_after=3,
            confidence_threshold=0.5,
            min_period=10,
            max_period=50,
            analysis_stride=1,
            analysis_size=64
        )
        
        loop_frames, before_frames, after_frames, loop_start, loop_period, confidence, report = results
        
        print("✓ 检测完成")
        print("\n" + "=" * 60)
        print("检测结果:")
        print("=" * 60)
        print(f"循环起始位置: {loop_start}")
        print(f"检测到的周期: {loop_period} 帧")
        print(f"真实周期: {true_loop_length} 帧")
        print(f"周期误差: {abs(loop_period - true_loop_length)} 帧")
        print(f"置信度: {confidence:.3f}")
        print(f"循环帧数量: {loop_frames.shape[0]}")
        print(f"前置帧数量: {before_frames.shape[0]}")
        print(f"后置帧数量: {after_frames.shape[0]}")
        
        # 验证结果
        print("\n" + "=" * 60)
        print("结果验证:")
        print("=" * 60)
        
        period_error = abs(loop_period - true_loop_length)
        if period_error <= 2:
            print(f"✓ 周期检测准确 (误差 {period_error} 帧)")
        else:
            print(f"⚠ 周期检测有偏差 (误差 {period_error} 帧)")
        
        if confidence >= 0.6:
            print(f"✓ 置信度良好 ({confidence:.3f})")
        elif confidence >= 0.4:
            print(f"⚠ 置信度中等 ({confidence:.3f})")
        else:
            print(f"✗ 置信度较低 ({confidence:.3f})")
        
        if loop_frames.shape[0] > 0:
            print(f"✓ 成功提取循环帧 ({loop_frames.shape[0]} 帧)")
        else:
            print("✗ 未提取到循环帧")
        
        # 打印详细报告
        print("\n" + "=" * 60)
        print("详细报告:")
        print("=" * 60)
        print(report)
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 检测失败: {e}")
        import traceback
        traceback.print_exc()


def test_edge_cases():
    """测试边界情况"""
    print("\n\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)
    
    from loop_detector import LoopDetectExtract
    node = LoopDetectExtract()
    
    # 测试1: 太短的视频
    print("\n测试 1: 视频帧数太少...")
    short_video = torch.rand(5, 32, 32, 3)
    try:
        results = node.detect_and_extract(
            frames=short_video,
            context_before=2,
            context_after=2,
            confidence_threshold=0.5,
        )
        print("✓ 正确处理短视频（返回全部帧）")
        print(f"  返回帧数: {results[0].shape[0]}")
    except Exception as e:
        print(f"✗ 处理短视频失败: {e}")
    
    # 测试2: 无循环的随机视频
    print("\n测试 2: 无循环的随机视频...")
    random_video = torch.rand(50, 32, 32, 3)
    try:
        results = node.detect_and_extract(
            frames=random_video,
            context_before=2,
            context_after=2,
            confidence_threshold=0.6,
        )
        confidence = results[5]
        print(f"✓ 正确处理无循环视频")
        print(f"  置信度: {confidence:.3f}")
        if confidence < 0.6:
            print("  ✓ 正确识别为低置信度")
    except Exception as e:
        print(f"✗ 处理随机视频失败: {e}")
    
    # 测试3: 完美循环
    print("\n测试 3: 完美循环（无噪声）...")
    perfect_video, true_period = create_synthetic_loop_video(
        loop_length=15,
        num_loops=4,
        noise_level=0.0
    )
    try:
        results = node.detect_and_extract(
            frames=perfect_video,
            context_before=3,
            context_after=3,
            confidence_threshold=0.5,
        )
        detected_period = results[4]
        confidence = results[5]
        print(f"✓ 检测完美循环")
        print(f"  真实周期: {true_period}")
        print(f"  检测周期: {detected_period}")
        print(f"  置信度: {confidence:.3f}")
    except Exception as e:
        print(f"✗ 检测完美循环失败: {e}")
    
    print("\n" + "=" * 60)
    print("边界测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行基础测试
    test_loop_detector()
    
    # 运行边界测试
    test_edge_cases()
    
    print("\n\n🎉 所有测试完成！")
    print("\n提示：在 ComfyUI 中使用时，节点会出现在 'video/analysis' 分类下")
    print("      节点名称: 🔁 循环检测与提取")

