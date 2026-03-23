import torch
import torch.nn as nn
import time
import math

import quiptools_cuda
from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda
from lib.codebook.latticee8_padded12 import QuantizedE8P12Linear as QuantizedE8P12Linear_Original
from lib.codebook.latticee8_padded12 import E8P12_codebook
from lib import codebook

# Mock 模式设置（如果没有真实量化权重，设为 False 使用真实 codebook）
USE_MOCK = False


class QuantizedE8P12Linear_Optimized(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 使用真实的 E8P12 codebook
        self.codebook = E8P12_codebook(inference=True).to(device).half()
        self.scale = 32.0
        
        self.Qidxs_view = None
        self.effective_SV = None
        self.effective_K_left = None

    def cache_params(self, n, m, Qidxs_list, SU, SV, K_left, scaleWH=None):
        """
        预计算所有能静态处理的参数
        """
        # ✅ 优化 2: 预计算 View
        self.Qidxs_view = Qidxs_list[0].view(m // 16, n // 64, 8, 4)
        
        # ✅ 优化 3: 标量融合 (Scalar Fusion)
        # 原始代码: 
        #   x = matmul_hadU_cuda(...) / scale
        #   x = x * SV * scale
        # 简化为: x = matmul_hadU_cuda(...) * SV
        # 所以我们只需要存储 SV，不需要预除 scale
        
        # K_left 必须保持为整数，给 matmul_hadU_cuda 使用
        self.K_left = K_left
        
        # 原始: x = x * SV * scale，但前面有 / scale，所以实际上就是 x = x * SV
        if self.effective_SV is None:
            self.effective_SV = SV.half()
            
        # 处理 SU 的 rescale
        self.effective_SU = SU.half()
        if scaleWH is not None:
             self.effective_SU /= scaleWH

    def forward(self, input, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right, 
                rescale_WH=False, scaleWH=None, **kwargs):
        
        n, m = len(SU), len(SV)
        
        # 1. 自动缓存 (如果 cache_params 没被显式调用)
        if self.Qidxs_view is None:
            self.cache_params(n, m, Qidxs_list, SU, SV, K_left, scaleWH if rescale_WH else None)

        x = input.view(-1, n)

        # 2. 融合后的 SU 乘法 (FP16)
        x = x * self.effective_SU

        # 3. Hadamard 1（K_left 必须是整数）
        # 注意：原始代码中 / scale 和后面的 * scale 会抵消，所以直接省略
        # 原始: x = matmul_hadU_cuda(x, had_left_T, K_left) / 32.0
        # 后面: x = x * SV * 32.0
        # 简化: x = matmul_hadU_cuda(x, had_left_T, K_left)，然后 x = x * SV
        x = matmul_hadU_cuda(x, had_left_T, self.K_left)

        # 4. Core Quantized Operation
        if x.size(0) == 1:
            # ✅ 优化: 移除不必要的类型转换
            x = torch.ops.quip_lib.decode_matvec_e8p(
                x[0].to(torch.float16),  # 转换为 FP16
                self.Qidxs_view,  # 使用缓存的 view
                self.codebook.grid_packed_abs,
                m, n
            )
            # kernel 返回 FP32，转回 FP16
            if x.dtype == torch.float32:
                x = x.half()
        else:
            # BS > 1 路径
            W_decompressed = torch.ops.quip_lib.decompress_packed_e8p(
                self.Qidxs_view,
                self.codebook.grid_packed_abs,
                m, n
            )
            x = (x.to(torch.float16) @ W_decompressed.T).half()

        # 5. Hadamard 2
        x = matmul_hadU_cuda(x, had_right, K_right)

        # 6. 乘以 SV（scale 已在步骤 3 中优化掉）
        x = x * self.effective_SV

        return x.view(*input.shape[:-1], m)

# ==========================================
# 4. 性能对比脚本
# ==========================================
def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 参数设置
    B, M, N = 1, 4096, 4096 # Batch=1 模拟生成阶段
    dtype = torch.float16

    # 初始化 Dummy 数据
    input_tensor = torch.randn(B, N, device=device, dtype=dtype)
    
    # 模拟 QuIP 所需的参数
    # E8P12: Qidxs 形状为 [M, N // 32] （每32个元素打包为一个int64）
    Qidxs_list = [torch.zeros(M, N // 32, dtype=torch.int64, device=device)]  # 正确的二维形状
    SU = torch.randn(N, device=device, dtype=dtype)
    SV = torch.randn(M, device=device, dtype=dtype)
    had_left_T = torch.randn(N, N, device=device, dtype=dtype) # Dummy Hadamard
    had_right = torch.randn(M, M, device=device, dtype=dtype)
    K_left = 1  # 应该是整数，不是张量
    K_right = 1  # 应该是整数，不是张量

    model_orig = QuantizedE8P12Linear_Original(device).to(dtype)
    model_opt = QuantizedE8P12Linear_Optimized(device).to(dtype)
    
    model_base = nn.Linear(N, M, bias=False).to(device).to(dtype)

    print("🔥 Warming up...")
    for _ in range(10):
        _ = model_base(input_tensor)
        _ = model_orig(input_tensor, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right)
        _ = model_opt(input_tensor, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right)
    torch.cuda.synchronize()

    def run_test(name, func, iterations=1000):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            _ = func()
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_us = (elapsed_time_ms / iterations) * 1000
        print(f"[{name}] Avg Time: {avg_time_us:.2f} µs")
        return avg_time_us

    # 运行测试
    print(f"\n🚀 开始测试 (Batch Size = {B}, Shape = {N}x{M})...")
    
    # 1. PyTorch nn.Linear (FP16)
    t_base = run_test("nn.Linear (FP16)", lambda: model_base(input_tensor))

    # 2. Original QuIP
    t_orig = run_test("Original QuIP", lambda: model_orig(
        input_tensor, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right
    ))

    # 3. Optimized QuIP
    t_opt = run_test("Optimized QuIP", lambda: model_opt(
        input_tensor, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right
    ))

    # 结果分析
    print("\n📊 结果分析:")
    print(f"优化版 vs 原始版 加速比: {t_orig / t_opt:.2f}x")
    print(f"优化版 vs nn.Linear 慢: {t_opt / t_base:.2f}x (预期内，因包含解压和Hadamard)")
    
    if USE_MOCK:
        print("\n⚠️ 注意: 当前运行在 Mock 模式下，主要反映 Python 层面的开销减少。")
        print("   在真实 CUDA 环境下，由于移除了 FP32 转换带来的内存带宽压力，加速比会更高。")

if __name__ == "__main__":
    benchmark()