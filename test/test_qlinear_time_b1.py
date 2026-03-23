import os
import sys
import torch
import time

# Add parent directory to path to import lib module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import gptq_data_utils
from lib.utils.unsafe_import import model_from_hf_path
from lib.linear import QuantizedLinear
torch.set_grad_enabled(False)

# 导入模型类
model, model_str = model_from_hf_path(
    "/fact_data/zeyuli/mixtral_8x7b_quip_full_noft",
    use_cuda_graph=False,
    use_flash_attn=False)


def benchmark_layer():
    device = "cuda"
    print("实例化模型层")
    linear = model.model.layers[0].self_attn.q_proj
    dummy_input = torch.randn(1, 1, 4096, device=device, dtype=torch.float16)

    _ = linear(dummy_input)
    torch.cuda.synchronize()

    print("正在预热 GPU...")
    linear.codebook_class.enable_profiling = False # 预热时不记录
    for _ in range(20):
        _ = linear(dummy_input)
    torch.cuda.synchronize()

    print("开始 Profiling...")
    linear.codebook_class.enable_profiling = True
    linear.codebook_class.kernel_timers = []
    linear.codebook_class.total_timers = []
    
    num_iters = 100
    with torch.no_grad():
        for _ in range(num_iters):
            _ = linear(dummy_input)
    
    torch.cuda.synchronize()
    
    print(f"收集到 {len(linear.codebook_class.total_timers)} 个 total timer 事件")
    print(f"收集到 {len(linear.codebook_class.kernel_timers)} 个 kernel timer 事件")
    
    if len(linear.codebook_class.total_timers) == 0:
        print("错误：没有收集到任何 timing 数据！")
        return

    pre_time = 0
    core_time = 0
    post_time = 0
    
    for (t0, t1, t2, t3) in linear.codebook_class.total_timers:
        pre_time += t0.elapsed_time(t1)
        core_time += t1.elapsed_time(t2)
        post_time += t2.elapsed_time(t3)

    avg_pre = pre_time / num_iters
    avg_core = core_time / num_iters
    avg_post = post_time / num_iters
    total = avg_pre + avg_core + avg_post

    print("\n" + "="*40)
    print(f"Batch Size = 1 性能瓶颈分析")
    print("="*40)
    print(f"测试迭代次数: {num_iters}")
    print(f"\n=== 分段耗时分析 ===")
    print(f"1. 前处理 (Hadamard + Scale): {avg_pre:.4f} ms ({avg_pre/total*100:.1f}%)")
    print(f"2. 核心 (Decode E8P):         {avg_core:.4f} ms ({avg_core/total*100:.1f}%)")
    print(f"3. 后处理 (Hadamard + Scale): {avg_post:.4f} ms ({avg_post/total*100:.1f}%)")
    print(f"总耗时:                       {total:.4f} ms")
    print("="*40)

    # 8. 自动给出建议
    core_ratio = (avg_core / total) * 100
    if core_ratio > 60:
        print(">> 结论：瓶颈在【核心 Decode 计算】。")
        print(">> 建议：必须重写 Triton Kernel，融合 Scale 和 Codebook 解压，利用 SRAM 减少 HBM 访问。")
    elif core_ratio < 20:
        print(">> 结论：瓶颈在【Hadamard 变换和前后处理】。")
        print(">> 建议：优化 Hadamard 变换实现或使用 CUDA Graph 减少开销。")
    else:
        print(">> 结论：混合型瓶颈。")
        print(">> 建议：Triton 算子融合收益最大，既能减少前后处理开销，又能优化显存访问。")

if __name__ == "__main__":
    benchmark_layer()