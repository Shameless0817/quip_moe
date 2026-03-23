"""
对比 QuantizedLinear 和普通 nn.Linear 的计算时间

分析在相同 input_size 和 output_size 下，两种层的性能差异
"""

import argparse
import time
import json

import torch
import torch.nn as nn
from tqdm import tqdm

from lib.linear.quantized_linear import QuantizedLinear
from lib import codebook


def benchmark_layer(layer, input_tensor, warmup=10, iterations=100, name="Layer"):
    """
    对单个层进行性能测试
    """
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(input_tensor)
        torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = layer(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations
    return avg_time


def create_quantized_linear(in_features, out_features, device='cuda'):
    """
    创建一个简单的 QuantizedLinear 层用于测试
    注意：由于 E8P12 的打包格式复杂，这个函数可能失败。
    建议使用 --theory-only 选项只做理论分析，或使用真实量化模型进行测试。
    """
    # E8P12 码本使用特殊的打包格式
    # 实际的 Qidxs 大小需要匹配 decompress_packed_e8p 的要求
    # 对于 (m, n) 的权重矩阵，打包后的 Qidxs 需要能 reshape 成 (m//16, n//64, 8, 4)
    
    # 检查维度是否满足要求
    if in_features % 64 != 0 or out_features % 16 != 0:
        raise ValueError(
            f"E8P12 要求 in_features 是 64 的倍数，out_features 是 16 的倍数。"
            f"当前: in_features={in_features}, out_features={out_features}"
        )
    
    # 注意：QuantizedLinear 中 codebook_id 设置为 7 (E8P12)
    # E8P12 参数: codesz=8, packsz=4, pack_out=False
    codesz, packsz = 8, 4
    
    # E8P12 使用 pack_out=False
    # Qidxs 形状: [out_features, in_features // (codesz * packsz)]
    ql = QuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        codesz=codesz,
        packsz=packsz,
        pack_out=False,  # E8P12 使用 pack_out=False
        idx_dtype='torch.int64',
        codebook_version=1,
        rank=0,
        rescale_WH=False,
        bias=False,
        resid_scale_override=-1,
        train_mode=False,
        grad_ckpt=False,
    ).to(device)
    
    # 手动初始化 Qidxs 为正确格式
    # E8P12: Qidxs 形状 [out_features, in_features // 32]
    packed_shape = (out_features, in_features // (codesz * packsz))
    ql.Qidxs = torch.randint(0, 256, packed_shape, dtype=torch.int64).to(device)
    
    # 触发一次前向传播以构建 codebook_class
    dummy_input = torch.randn(1, in_features).to(device)
    try:
        _ = ql(dummy_input)
    except Exception as e:
        raise RuntimeError(
            f"QuantizedLinear 前向传播失败: {e}\n"
            f"这通常是因为 Qidxs 的打包格式不匹配。\n"
            f"建议使用 --theory-only 选项跳过实际测试。"
        )
    
    return ql


def analyze_computational_complexity(in_features, out_features, batch_size):
    """
    理论计算复杂度分析
    """
    print("\n" + "="*80)
    print("理论计算复杂度分析")
    print("="*80)
    
    # 普通 Linear: y = x @ W.T + b
    # 计算量: batch_size * in_features * out_features 次乘加操作
    linear_ops = batch_size * in_features * out_features * 2  # 乘法+加法
    linear_flops = linear_ops
    
    print(f"\n【普通 Linear 层】")
    print(f"  操作: y = x @ W.T")
    print(f"  矩阵乘法: ({batch_size}, {in_features}) @ ({out_features}, {in_features}).T")
    print(f"  FLOPs: {linear_flops:,} ({linear_flops/1e9:.2f} GFLOPs)")
    
    # QuantizedLinear 的计算流程:
    # 1. x = x * SU  (element-wise)
    # 2. x = matmul_hadU_cuda(x, had_left_T, K_left)  (Hadamard变换)
    # 3. x = decompress + matmul  (解压量化权重 + 矩阵乘法)
    # 4. x = matmul_hadU_cuda(x, had_right, K_right)  (Hadamard变换)
    # 5. x = x * SV  (element-wise)
    
    print(f"\n【QuantizedLinear 层】")
    print(f"  操作流程:")
    print(f"    1. x = x * SU  (逐元素乘法)")
    print(f"       FLOPs: {batch_size * in_features:,}")
    
    print(f"    2. x = Hadamard_left(x)")
    print(f"       复杂度: O(n log n) 快速Hadamard变换")
    hadamard_left_ops = batch_size * in_features * int(torch.log2(torch.tensor(in_features)).item())
    print(f"       约 {hadamard_left_ops:,} ops")
    
    print(f"    3. 解压 + 矩阵乘法")
    print(f"       解压: 从量化索引查找码本 (内存访问密集)")
    print(f"       矩阵乘法: ({batch_size}, {in_features}) @ ({out_features}, {in_features}).T")
    decompress_matmul_ops = linear_ops  # 矩阵乘法部分相同
    print(f"       FLOPs (仅矩阵乘法): {decompress_matmul_ops:,}")
    
    print(f"    4. x = Hadamard_right(x)")
    hadamard_right_ops = batch_size * out_features * int(torch.log2(torch.tensor(out_features)).item())
    print(f"       约 {hadamard_right_ops:,} ops")
    
    print(f"    5. x = x * SV  (逐元素乘法)")
    print(f"       FLOPs: {batch_size * out_features:,}")
    
    quant_total_ops = (batch_size * in_features + 
                       hadamard_left_ops + 
                       decompress_matmul_ops + 
                       hadamard_right_ops + 
                       batch_size * out_features)
    
    print(f"\n  总 FLOPs (不含解压开销): {quant_total_ops:,} ({quant_total_ops/1e9:.2f} GFLOPs)")
    print(f"  额外开销: 量化权重解压 (内存访问)")
    
    print(f"\n【对比】")
    print(f"  计算量比例 (Quantized/Linear): {quant_total_ops/linear_flops:.2f}x")
    print(f"  注意: QuantizedLinear 的主要开销在于:")
    print(f"    - 权重解压 (内存访问)")
    print(f"    - Hadamard 变换 (额外计算)")
    print(f"    - 但权重占用内存大幅减少 (2-bit vs 16-bit, 8x压缩)")


def run_benchmark(args):
    """
    运行性能对比测试
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"批次大小: {args.batch_size}")
    print(f"维度配置: {args.configs}")
    
    configs = []
    for config_str in args.configs.split(';'):
        in_feat, out_feat = map(int, config_str.split(','))
        configs.append((in_feat, out_feat))
    
    results = []
    
    for in_features, out_features in tqdm(configs, desc="测试不同维度配置"):
        print(f"\n" + "="*80)
        print(f"测试配置: in_features={in_features}, out_features={out_features}")
        print("="*80)
        
        # 创建输入
        input_tensor = torch.randn(args.batch_size, in_features).to(device)
        
        # 创建普通 Linear 层
        print("\n创建普通 Linear 层...")
        linear = nn.Linear(in_features, out_features, bias=False).to(device)
        
        # 创建 QuantizedLinear 层（如果需要）
        quantized = None
        if not args.theory_only:
            print("创建 QuantizedLinear 层...")
            quantized = create_quantized_linear(in_features, out_features, device)
        
        # Benchmark Linear
        print(f"\n测试普通 Linear 层...")
        linear_time = benchmark_layer(
            linear, input_tensor, 
            warmup=args.warmup, 
            iterations=args.iterations,
            name="Linear"
        )
        
        # Benchmark QuantizedLinear（如果可用）
        quantized_time = None
        if quantized is not None:
            print(f"测试 QuantizedLinear 层...")
            quantized_time = benchmark_layer(
                quantized, input_tensor,
                warmup=args.warmup,
                iterations=args.iterations, 
                name="QuantizedLinear"
            )
        else:
            print(f"跳过 QuantizedLinear 测试（使用理论估算）")
            # 理论估算：QuantizedLinear 通常慢 3-5x
            quantized_time = linear_time * 4.0  # 使用 4x 作为估算
        
        # 计算内存占用
        linear_memory = (in_features * out_features * 2) / (1024**2)  # FP16, MB
        
        # QuantizedLinear 内存组成:
        # 1. Qidxs 索引: out_features * (in_features / (codesz * packsz)) * sizeof(int64)
        #    codesz=8, packsz=4, 所以是 out_features * (in_features / 32) * 8 bytes
        # 2. SU, SV 缩放因子: (in_features + out_features) * 2 bytes (FP16)
        # 3. 实际量化权重是 2-bit per weight
        codesz, packsz = 8, 4
        qidxs_memory = (out_features * (in_features / (codesz * packsz)) * 8) / (1024**2)  # int64 索引
        scales_memory = (in_features + out_features) * 2 / (1024**2)  # SU, SV
        quantized_memory = qidxs_memory + scales_memory
        
        speedup = linear_time / quantized_time
        memory_ratio = quantized_memory / linear_memory
        
        result = {
            'in_features': in_features,
            'out_features': out_features,
            'batch_size': args.batch_size,
            'linear_time_ms': linear_time * 1000,
            'quantized_time_ms': quantized_time * 1000,
            'speedup': speedup,
            'linear_memory_mb': linear_memory,
            'quantized_memory_mb': quantized_memory,
            'memory_ratio': memory_ratio,
            'quantized_tested': quantized is not None,  # 标记是否实际测试了
        }
        results.append(result)
        
        # 打印结果
        print(f"\n{'='*80}")
        print(f"性能对比结果")
        print(f"{'='*80}")
        print(f"  普通 Linear:")
        print(f"    时间: {linear_time*1000:.3f} ms")
        print(f"    内存: {linear_memory:.2f} MB (FP16 权重)")
        print(f"\n  QuantizedLinear:")
        if quantized is not None:
            print(f"    时间: {quantized_time*1000:.3f} ms (实测)")
        else:
            print(f"    时间: {quantized_time*1000:.3f} ms (理论估算: ~4x Linear)")
        print(f"    内存: {quantized_memory:.2f} MB (约 2-bit 权重)")
        print(f"\n  对比:")
        if speedup > 1:
            print(f"    速度: QuantizedLinear 慢 {speedup:.2f}x")
        else:
            print(f"    速度: QuantizedLinear 快 {1/speedup:.2f}x")
        print(f"    内存: QuantizedLinear 节省 {(1-memory_ratio)*100:.1f}% ({1/memory_ratio:.2f}x 压缩)")
        print(f"    权衡: 内存节省 vs 计算开销")
    
    # 打印汇总表格
    if results:
        print("\n" + "="*100)
        print("汇总表格")
        print("="*100)
        print(f"{'维度':<20} {'Linear (ms)':<15} {'Quantized (ms)':<15} {'速度比':<12} {'内存比':<12}")
        print("-"*100)
        for r in results:
            dim_str = f"{r['in_features']}x{r['out_features']}"
            speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] > 1 else f"{1/r['speedup']:.2f}x (快)"
            print(f"{dim_str:<20} {r['linear_time_ms']:<15.3f} {r['quantized_time_ms']:<15.3f} "
                  f"{speedup_str:<12} {r['memory_ratio']:<12.2f}")
        print("="*100)
        print("\n注: 速度比 > 1 表示 QuantizedLinear 更慢; 内存比 < 1 表示内存节省")
    
    # 保存结果
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({
                'batch_size': args.batch_size,
                'warmup': args.warmup,
                'iterations': args.iterations,
                'results': results
            }, f, indent=2)
        print(f"\n结果已保存到: {args.output_json}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='对比 QuantizedLinear 和普通 Linear 层的性能'
    )
    parser.add_argument(
        '--configs',
        type=str,
        default='4096,4096;4096,11008;11008,4096',
        help='维度配置，格式: in1,out1;in2,out2;...'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='批次大小'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='预热迭代次数'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='测试迭代次数'
    )
    parser.add_argument(
        '--show_analysis',
        action='store_true',
        help='显示理论计算复杂度分析'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='保存结果到 JSON 文件'
    )
    parser.add_argument(
        '--theory_only',
        action='store_true',
        help='仅进行理论分析，不实际创建和测试 QuantizedLinear 层'
    )
    
    args = parser.parse_args()
    
    print("="*100)
    print("QuantizedLinear vs Linear 性能对比测试")
    print("="*100)
    
    run_benchmark(args)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()
