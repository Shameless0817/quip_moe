"""
诊断 SU/SV 在量化过程中产生异常值的原因

这个脚本会：
1. 加载 Mixtral 模型的一个 linear 层
2. 逐步执行量化过程
3. 监控每一步的 SU/SV 值
4. 分析异常值产生的具体位置和原因
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, '/fact_home/zeyuli/quip_sharp')

from transformers import AutoModelForCausalLM, AutoTokenizer


def analyze_tensor_stats(tensor, name):
    """分析张量的统计信息"""
    print(f"\n{'='*60}")
    print(f"分析: {name}")
    print(f"{'='*60}")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Min: {tensor.min().item():.6e}")
    print(f"Max: {tensor.max().item():.6e}")
    print(f"Mean: {tensor.mean().item():.6e}")
    print(f"Std: {tensor.std().item():.6e}")
    
    # 检查异常值
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    has_zero = (tensor == 0).any().item()
    
    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")
    print(f"Has Zero: {has_zero}")
    
    if has_nan:
        nan_count = torch.isnan(tensor).sum().item()
        print(f"  NaN count: {nan_count} / {tensor.numel()} ({100*nan_count/tensor.numel():.2f}%)")
    
    if has_inf:
        inf_count = torch.isinf(tensor).sum().item()
        print(f"  Inf count: {inf_count} / {tensor.numel()} ({100*inf_count/tensor.numel():.2f}%)")
    
    # 值的分布
    abs_tensor = tensor.abs()
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"\n值的百分位分布 (绝对值):")
    for p in percentiles:
        val = torch.quantile(abs_tensor, p/100.0).item()
        print(f"  {p:2d}%: {val:.6e}")
    
    return has_nan or has_inf


def manual_quantize_linear(linear, codebook, hessian=None, device='cuda'):
    """
    手动执行量化过程，监控每一步的 SU/SV 值
    
    这是 lib.algo.quip.quantize_linear 的诊断版本
    """
    print("\n" + "="*80)
    print("开始手动量化过程")
    print("="*80)
    
    W = linear.weight.data.clone().to(device)
    m, n = W.shape
    
    print(f"\n权重矩阵形状: {W.shape}")
    analyze_tensor_stats(W, "原始权重 W")
    
    # Step 1: 随机 Hadamard 变换预处理
    print("\n" + "-"*80)
    print("Step 1: Hadamard 预处理 (incoherence_preprocess)")
    print("-"*80)
    
    from lib.algo import quip
    
    # 初始化 SU, SV (应该是 ±1)
    SU = (torch.randn(n, device=device).sign() + 1e-5).sign()
    SV = (torch.randn(m, device=device).sign() + 1e-5).sign()
    
    print(f"\n初始 SU (随机 ±1):")
    analyze_tensor_stats(SU, "初始 SU")
    print(f"\n初始 SV (随机 ±1):")
    analyze_tensor_stats(SV, "初始 SV")
    
    # Hadamard 变换
    if hessian is None:
        print("\n警告: 没有提供 Hessian 矩阵，使用单位矩阵")
        H = torch.eye(n, device=device)
    else:
        H = hessian.to(device)
    
    analyze_tensor_stats(H.diag(), "Hessian 对角线")
    
    # 调用 incoherence_preprocess
    result = quip.incoherence_preprocess(W, H, SU, SV, 
                                         had_left=True, 
                                         had_right=True,
                                         K_left=3,
                                         K_right=3)
    
    if result is None:
        print("\n❌ incoherence_preprocess 返回 None - 这是第一个可能产生异常的地方!")
        print("原因: Hessian 矩阵的 Cholesky 分解失败")
        return None, None, None
    
    Wq, Hq, SU_new, SV_new = result
    
    print(f"\n✓ incoherence_preprocess 成功")
    
    has_invalid_su = analyze_tensor_stats(SU_new, "预处理后的 SU")
    has_invalid_sv = analyze_tensor_stats(SV_new, "预处理后的 SV")
    
    if has_invalid_su or has_invalid_sv:
        print("\n⚠️  警告: 预处理后就已经出现 invalid SU/SV!")
        print("可能原因:")
        print("1. Hessian 矩阵条件数过大 (ill-conditioned)")
        print("2. Cholesky 分解产生数值不稳定")
        print("3. 权重矩阵范围过大导致中间计算溢出")
    
    analyze_tensor_stats(Wq, "预处理后的权重 Wq")
    analyze_tensor_stats(Hq.diag(), "预处理后的 Hessian 对角线")
    
    # Step 2: 量化
    print("\n" + "-"*80)
    print("Step 2: 向量量化")
    print("-"*80)
    
    # LDLQ 量化
    from lib.algo.ldlq import ldlq
    
    # 检查条件数
    hess_diag = Hq.diag()
    cond_number = hess_diag.max() / (hess_diag.min() + 1e-10)
    print(f"\nHessian 条件数: {cond_number:.2e}")
    if cond_number > 1e6:
        print("⚠️  警告: Hessian 条件数过大，可能导致数值不稳定")
    
    try:
        Qidxs = ldlq(Wq, Hq, codebook)
        print(f"\n✓ 量化成功, Qidxs shape: {Qidxs.shape}")
        analyze_tensor_stats(Qidxs.float(), "量化索引 Qidxs")
    except Exception as e:
        print(f"\n❌ 量化失败: {e}")
        return None, None, None
    
    # Step 3: 重构权重并检查误差
    print("\n" + "-"*80)
    print("Step 3: 重构权重")
    print("-"*80)
    
    # 从 codebook 重构
    W_reconstructed = codebook[Qidxs.flatten()].reshape(Wq.shape)
    analyze_tensor_stats(W_reconstructed, "重构后的权重")
    
    # 计算量化误差
    quant_error = (Wq - W_reconstructed).abs()
    analyze_tensor_stats(quant_error, "量化误差")
    
    # 相对误差
    rel_error = quant_error / (Wq.abs() + 1e-10)
    analyze_tensor_stats(rel_error, "相对量化误差")
    
    print(f"\n量化误差统计:")
    print(f"  最大绝对误差: {quant_error.max().item():.6e}")
    print(f"  平均绝对误差: {quant_error.mean().item():.6e}")
    print(f"  最大相对误差: {rel_error.max().item():.6e}")
    print(f"  平均相对误差: {rel_error.mean().item():.6e}")
    
    return SU_new, SV_new, Qidxs


def diagnose_layer_quantization(model_name, layer_name, device='cuda'):
    """
    诊断特定层的量化过程
    """
    print("="*80)
    print(f"诊断层: {layer_name}")
    print("="*80)
    
    # 加载模型
    print(f"\n加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    
    # 获取指定层
    layer = model
    for attr in layer_name.split('.'):
        layer = getattr(layer, attr)
    
    if not isinstance(layer, nn.Linear):
        print(f"错误: {layer_name} 不是 Linear 层")
        return
    
    print(f"\nLinear 层信息:")
    print(f"  输入维度: {layer.in_features}")
    print(f"  输出维度: {layer.out_features}")
    print(f"  有偏置: {layer.bias is not None}")
    
    # 移到设备
    layer = layer.to(device)
    
    # 准备 codebook
    print(f"\n准备 codebook...")
    from lib.codebook.codebook_cuda import get_codebook
    codebook_id = 'E8P12'
    cb_cuda, _ = get_codebook(codebook_id)
    codebook = cb_cuda.codebook_to_cpu()
    print(f"Codebook: {codebook_id}, shape: {codebook.shape}")
    
    # 尝试量化
    SU, SV, Qidxs = manual_quantize_linear(layer, codebook, device=device)
    
    if SU is None:
        print("\n" + "="*80)
        print("总结: 量化失败")
        print("="*80)
        print("主要原因: Hessian 矩阵预处理失败")
        print("\n可能的解决方案:")
        print("1. 使用实际的 Hessian 矩阵 (从数据计算)")
        print("2. 添加 --sigma_reg 正则化参数")
        print("3. 使用 --rescale_WH 选项")
        return
    
    # 最终分析
    print("\n" + "="*80)
    print("总结: SU/SV 异常值分析")
    print("="*80)
    
    has_invalid_su = torch.isnan(SU).any() or torch.isinf(SU).any()
    has_invalid_sv = torch.isnan(SV).any() or torch.isinf(SV).any()
    
    if has_invalid_su or has_invalid_sv:
        print("\n❌ 发现 invalid SU/SV!")
        print("\n根本原因分析:")
        print("-" * 60)
        
        if has_invalid_su:
            print("\n1. SU 异常:")
            print(f"   NaN 数量: {torch.isnan(SU).sum().item()}")
            print(f"   Inf 数量: {torch.isinf(SU).sum().item()}")
        
        if has_invalid_sv:
            print("\n2. SV 异常:")
            print(f"   NaN 数量: {torch.isnan(SV).sum().item()}")
            print(f"   Inf 数量: {torch.isinf(SV).sum().item()}")
        
        print("\n产生原因:")
        print("=" * 60)
        print("""
1. **Hessian 矩阵问题**
   - 没有使用真实 Hessian (我们用的是单位矩阵)
   - Hessian 条件数过大 (ill-conditioned)
   - Cholesky 分解失败或数值不稳定

2. **权重范围问题**
   - 权重值范围过大 (检查上面的权重统计)
   - 与 Hessian 组合时产生数值溢出
   - float16 精度不足以表示中间结果

3. **数值稳定性**
   - Hadamard 变换的累积误差
   - 矩阵分解的舍入误差
   - float16 vs float32/64 的精度差异

解决方案:
-----------
1. 使用真实 Hessian 矩阵 (需要先计算):
   python save_hessian_mixtral.sh

2. 添加正则化:
   --sigma_reg 1e-2  # 增加 Hessian 对角线的稳定性

3. 启用权重缩放:
   --rescale_WH  # 平衡权重和 Hessian 的范围

4. 使用更高精度:
   --use_fp64  # 在关键步骤使用 float64
        """)
    else:
        print("\n✓ SU/SV 正常!")
        print("但注意: 我们没有使用真实的 Hessian 矩阵")
        print("在实际量化中仍可能出现问题")


def test_with_real_hessian(hessian_path, layer_name, device='cuda'):
    """
    使用真实 Hessian 测试
    """
    print("\n" + "="*80)
    print("使用真实 Hessian 重新测试")
    print("="*80)
    
    # 加载 Hessian
    print(f"\n加载 Hessian: {hessian_path}")
    hess_data = torch.load(hessian_path, map_location='cpu')
    
    if layer_name not in hess_data:
        print(f"错误: {layer_name} 不在 Hessian 文件中")
        print(f"可用的层: {list(hess_data.keys())[:10]}...")
        return
    
    H = hess_data[layer_name]['H']
    print(f"Hessian shape: {H.shape}")
    analyze_tensor_stats(H.diag(), "真实 Hessian 对角线")
    
    # 检查条件数
    H_diag = H.diag()
    cond = H_diag.max() / (H_diag.min() + 1e-10)
    print(f"\nHessian 条件数: {cond:.2e}")
    
    if cond > 1e8:
        print("⚠️  警告: Hessian 条件数极大，建议使用 --sigma_reg")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=str, 
                        default='model.layers.0.self_attn.q_proj',
                        help='要诊断的层名称')
    parser.add_argument('--hessian', type=str, default=None,
                        help='Hessian 文件路径 (可选)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    print("SU/SV 量化诊断工具")
    print("="*80)
    print(f"层: {args.layer}")
    print(f"设备: {args.device}")
    print("="*80)
    
    # 直接使用模拟的权重矩阵进行测试，不加载完整模型
    print("\n使用模拟权重进行诊断 (避免内存溢出)...")
    print("="*80)
    
    # 创建模拟的 linear 层 (4096x4096, 类似 Mixtral)
    device = args.device
    torch.manual_seed(42)
    
    class FakeLinear(nn.Linear):
        def __init__(self):
            super().__init__(4096, 4096, bias=False)
            # 使用类似 Mixtral 的权重初始化
            nn.init.normal_(self.weight, mean=0, std=0.02)
    
    layer = FakeLinear().to(device)
    
    print(f"\n模拟 Linear 层信息:")
    print(f"  输入维度: {layer.in_features}")
    print(f"  输出维度: {layer.out_features}")
    print(f"  权重形状: {layer.weight.shape}")
    
    # 准备 codebook
    print(f"\n准备 codebook...")
    from lib.codebook.codebook_cuda import get_codebook
    codebook_id = 'E8P12'
    cb_cuda, _ = get_codebook(codebook_id)
    codebook = cb_cuda.codebook_to_cpu()
    print(f"Codebook: {codebook_id}, shape: {codebook.shape}")
    
    # 测试不同的 Hessian 配置
    print("\n" + "="*80)
    print("测试1: 使用单位矩阵作为 Hessian (最简单情况)")
    print("="*80)
    
    SU, SV, Qidxs = manual_quantize_linear(layer, codebook, hessian=None, device=device)
    
    if SU is not None:
        has_invalid = torch.isnan(SU).any() or torch.isinf(SU).any() or torch.isnan(SV).any() or torch.isinf(SV).any()
        if has_invalid:
            print("\n❌ 即使使用单位 Hessian 也出现异常值!")
            print("说明问题出在 Hadamard 变换或权重范围本身")
        else:
            print("\n✓ 使用单位 Hessian 没有异常值")
    
    # 测试2: 使用条件数较大的 Hessian
    print("\n" + "="*80)
    print("测试2: 使用条件数较大的 Hessian (模拟真实情况)")
    print("="*80)
    
    # 创建一个条件数较大的 Hessian
    torch.manual_seed(123)
    H_bad = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    H_bad = H_bad @ H_bad.T
    # 让某些特征值特别小
    eigenvalues = torch.linspace(1e-6, 1.0, 4096, device=device)
    H_bad = H_bad * eigenvalues.view(-1, 1) * eigenvalues.view(1, -1)
    H_bad = H_bad + torch.eye(4096, device=device) * 1e-7
    
    cond_bad = torch.linalg.cond(H_bad).item()
    print(f"Hessian 条件数: {cond_bad:.2e}")
    
    SU2, SV2, Qidxs2 = manual_quantize_linear(layer, codebook, hessian=H_bad, device=device)
    
    if SU2 is not None:
        has_invalid = torch.isnan(SU2).any() or torch.isinf(SU2).any() or torch.isnan(SV2).any() or torch.isinf(SV2).any()
        if has_invalid:
            print("\n❌ 使用病态 Hessian 导致异常值!")
            print("这是主要原因: Hessian 矩阵条件数过大")
        else:
            print("\n✓ 即使使用病态 Hessian 也没有异常值 (运气好)")
    
    # 如果有真实 Hessian，测试
    if args.hessian:
        print("\n" + "="*80)
        print("测试3: 使用真实 Hessian")
        print("="*80)
        test_with_real_hessian(args.hessian, args.layer, args.device)
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
