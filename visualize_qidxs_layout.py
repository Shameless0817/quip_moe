"""
可视化 E8P12 Qidxs 的 bit 排列

运行此脚本可以理解一个 64-bit int64 如何编码 4 个码字（32 个元素）
"""

import torch
import numpy as np

def visualize_single_int64():
    """可视化单个 int64 的 bit 布局"""
    print("=" * 80)
    print("单个 int64 (64 bits) 的布局")
    print("=" * 80)
    print()
    
    # 创建一个示例 int64
    # 假设 4 个码字索引分别是: 0x1234, 0x5678, 0x9ABC, 0xDEF0
    codeword_0 = 0x1234  # 16 bits: abs=0x12, sign=0x34
    codeword_1 = 0x5678  # 16 bits: abs=0x56, sign=0x78
    codeword_2 = 0x9ABC  # 16 bits: abs=0x9A, sign=0xBC
    codeword_3 = 0xDEF0  # 16 bits: abs=0xDE, sign=0xF0
    
    # 提取 abs 和 sign 部分
    abs_0 = (codeword_0 >> 8) & 0xFF
    abs_1 = (codeword_1 >> 8) & 0xFF
    abs_2 = (codeword_2 >> 8) & 0xFF
    abs_3 = (codeword_3 >> 8) & 0xFF
    
    sign_0 = codeword_0 & 0xFF
    sign_1 = codeword_1 & 0xFF
    sign_2 = codeword_2 & 0xFF
    sign_3 = codeword_3 & 0xFF
    
    # 构建 abs32
    abs32 = abs_0 | (abs_1 << 8) | (abs_2 << 16) | (abs_3 << 24)
    
    # 构建 sign32 (交错排列)
    sign32 = 0
    for i in range(4):  # 4 个码字
        sign_bits = [sign_0, sign_1, sign_2, sign_3][i]
        for j in range(8):  # 每个码字的 8 个 sign bits
            if (sign_bits >> j) & 1:
                sign32 |= (1 << (4 * j + i))
    
    # 组合成 64-bit
    packed_int64 = (sign32 << 32) | abs32
    
    print(f"输入的 4 个码字索引 (16-bit 每个):")
    print(f"  码字 0: 0x{codeword_0:04X} (abs=0x{abs_0:02X}, sign=0x{sign_0:02X})")
    print(f"  码字 1: 0x{codeword_1:04X} (abs=0x{abs_1:02X}, sign=0x{sign_1:02X})")
    print(f"  码字 2: 0x{codeword_2:04X} (abs=0x{abs_2:02X}, sign=0x{sign_2:02X})")
    print(f"  码字 3: 0x{codeword_3:04X} (abs=0x{abs_3:02X}, sign=0x{sign_3:02X})")
    print()
    
    print(f"打包后的 64-bit int64: 0x{packed_int64:016X}")
    print()
    
    print("Bit 布局:")
    print(f"  Bits [31:0]  (abs32):  0x{abs32:08X}")
    print(f"    Bits [7:0]   = 0x{abs_0:02X}  (码字 0 的 abs 索引)")
    print(f"    Bits [15:8]  = 0x{abs_1:02X}  (码字 1 的 abs 索引)")
    print(f"    Bits [23:16] = 0x{abs_2:02X}  (码字 2 的 abs 索引)")
    print(f"    Bits [31:24] = 0x{abs_3:02X}  (码字 3 的 abs 索引)")
    print()
    print(f"  Bits [63:32] (sign32): 0x{sign32:08X}")
    print(f"    码字 0 的 8 个 sign bits 位于: bits [32, 36, 40, 44, 48, 52, 56, 60]")
    print(f"    码字 1 的 8 个 sign bits 位于: bits [33, 37, 41, 45, 49, 53, 57, 61]")
    print(f"    码字 2 的 8 个 sign bits 位于: bits [34, 38, 42, 46, 50, 54, 58, 62]")
    print(f"    码字 3 的 8 个 sign bits 位于: bits [35, 39, 43, 47, 51, 55, 59, 63]")
    print()
    
    # 详细展示 sign bits 的交错
    print("Sign bits 详细布局 (每 4 bits 一组):")
    for nibble_idx in range(8):  # 8 个 4-bit 组
        start_bit = 32 + nibble_idx * 4
        nibble_val = (sign32 >> (nibble_idx * 4)) & 0xF
        print(f"  Bits [{start_bit+3}:{start_bit}] = 0b{nibble_val:04b} = ", end="")
        print(f"[码字3.bit{nibble_idx}={nibble_val>>3&1}, "
              f"码字2.bit{nibble_idx}={nibble_val>>2&1}, "
              f"码字1.bit{nibble_idx}={nibble_val>>1&1}, "
              f"码字0.bit{nibble_idx}={nibble_val&1}]")
    print()
    
    return packed_int64


def visualize_shape_transform():
    """可视化形状变换过程"""
    print("=" * 80)
    print("形状变换流程 (以 4096x4096 矩阵为例)")
    print("=" * 80)
    print()
    
    m, n = 4096, 4096
    
    print(f"1. 原始权重矩阵: [{m}, {n}] float16")
    print(f"   总元素数: {m * n:,}")
    print()
    
    print(f"2. 量化后 (LDLQ): [{m}, {n // 8}] int64")
    print(f"   总索引数: {m * n // 8:,}")
    print(f"   每个索引: 16 bits (量化 8 个元素)")
    print(f"   验证: {m * n // 8} 索引 × 8 元素/索引 = {m * n:,} 元素 ✓")
    print()
    
    print(f"3. 打包后 (maybe_pack_idxs): [{m}, {n // 32}] int64")
    print(f"   总 int64 数: {m * n // 32:,}")
    print(f"   每个 int64: 64 bits (打包 4 个 16-bit 索引)")
    print(f"   验证: {m * n // 32} int64 × 4 码字/int64 × 8 元素/码字 = {m * n:,} 元素 ✓")
    print()
    
    print(f"4. Reshape 给 CUDA kernel: [{m // 16}, {n // 64}, 8, 4] int64")
    print(f"   = [{m // 16}, {n // 64}, 8, 4]")
    print(f"   维度解释:")
    print(f"     - 第1维 ({m // 16}): 每 16 行一组 (CUDA block 处理单位)")
    print(f"     - 第2维 ({n // 64}): 每 64 列一组 (每组 2 个 warp × 32 列)")
    print(f"     - 第3维 (8): 每组 8 个 int64")
    print(f"     - 第4维 (4): 每个 int64 包含 4 个码字")
    print(f"   验证: {m // 16} × {n // 64} × 8 × 4 = {(m // 16) * (n // 64) * 8 * 4:,} = {m * n // 32:,} ✓")
    print()
    
    print(f"5. 解压缩后: [{m}, {n}] float16")
    print(f"   还原到原始形状")
    print()


def demonstrate_packing():
    """演示实际的打包过程"""
    print("=" * 80)
    print("实际打包演示")
    print("=" * 80)
    print()
    
    # 创建一个小的示例，必须满足 E8P 的维度要求
    # m 必须是 16 的倍数，n 必须是 64 的倍数
    m, n = 32, 256  
    
    # 模拟量化索引 [m, n//8] (每 8 个元素一个 16-bit 索引)
    torch.manual_seed(42)
    idxs = torch.randint(0, 65536, (m, n // 8), dtype=torch.int64)  # [32, 32]
    
    print(f"原始量化索引形状: {list(idxs.shape)} = [{m}, {n // 8}]")
    print(f"  (每个索引编码 8 个元素)")
    print(f"示例数据 (前 2 行, 前 4 列):")
    print(idxs[:2, :4])
    print()
    
    # 按照 maybe_pack_idxs 的逻辑打包
    # 注意: maybe_pack_idxs 中 idxs 的形状是 [m, n]，其中 n 已经是 original_n // 8
    # 所以 (n * 8) // 16 = n // 2
    idxs_view = idxs.view(m // 2, 2, (n // 8) // 2, 2).transpose(1, 2).contiguous()
    print(f"重排后形状: {list(idxs_view.shape)} = [{m // 2}, {(n // 8) // 2}, 2, 2]")
    print(f"  (准备提取 4 个码字的 abs 和 sign)")
    print()
    
    # 步骤 2: 提取 abs 部分 (每个索引的高 8 bits)
    abs32 = (idxs_view[:, :, 0, 0] >> 8) + \
            ((idxs_view[:, :, 1, 0] >> 8) << 8) + \
            ((idxs_view[:, :, 0, 1] >> 8) << 16) + \
            ((idxs_view[:, :, 1, 1] >> 8) << 24)
    
    print(f"abs32 形状: {list(abs32.shape)} = [{m // 2}, {(n // 8) // 2}]")
    print(f"  (将 4 个 8-bit abs 索引打包成 32 bits)")
    print(f"abs32 示例 (第 0 行, 第 0 列，十六进制): 0x{abs32[0, 0].item():08X}")
    print()
    
    # 步骤 3: 提取 sign 部分 (每个索引的低 8 bits，交错排列)
    sign32 = torch.zeros_like(abs32)
    for i in range(4):  # 4 个码字
        wt = idxs_view[:, :, i % 2, i // 2]
        for j in range(8):  # 每个码字的 8 个 sign bits
            sign32 += ((wt >> j) & 1) << (4 * j + i)
    
    print(f"sign32 形状: {list(sign32.shape)} = [{m // 2}, {(n // 8) // 2}]")
    print(f"  (8 个 sign bits × 4 个码字 = 32 bits，交错排列)")
    print(f"sign32 示例 (第 0 行, 第 0 列，十六进制): 0x{sign32[0, 0].item():08X}")
    print()
    
    # 步骤 4: 组合成 64-bit int64
    packed_temp = (sign32.to(torch.int64) << 32) | abs32.to(torch.int64)
    print(f"组合后临时形状: {list(packed_temp.shape)} = [{m // 2}, {(n // 8) // 2}]")
    print(f"  (每个元素是 64-bit int64)")
    print()
    
    # 步骤 5: Reshape 到最终形状
    # 按照原始代码: output.reshape(m // 16, 8, n // 8, 4).transpose(1, 2)
    # 其中 n 是 idxs的第二维，也就是我们的 n // 8
    n_packed_idx = n // 8  # idxs 的第二维
    packed = packed_temp.reshape(m // 16, 8, n_packed_idx // 8, 4).transpose(1, 2).contiguous()
    packed = packed.view(m, n_packed_idx // 4)  # 最终是 [m, (n//8)//4] = [m, n//32]
    
    print(f"最终打包形状: {list(packed.shape)} = [{m}, {n // 32}]")
    print(f"  (从 [{m}, {n // 8}] 量化索引 → [{m}, {n // 32}] 打包索引)")
    print(f"  (压缩比: 4个 16-bit 索引 → 1个 64-bit int64)")
    print(f"示例数据 (前 2 行, 前 2 列，十六进制):")
    for i in range(2):
        for j in range(min(2, n // 32)):
            val = packed[i,j].item()
            print(f"  packed[{i},{j}] = 0x{val:016X}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("E8P12 Qidxs Bit 排列可视化工具")
    print("=" * 80 + "\n")
    
    # 1. 单个 int64 的布局
    visualize_single_int64()
    print("\n")
    
    # 2. 形状变换
    visualize_shape_transform()
    print("\n")
    
    # 3. 实际打包演示
    demonstrate_packing()
    
    print("=" * 80)
    print("总结:")
    print("=" * 80)
    print("• 每个 int64 编码 4 个码字 (每个码字 16 bits)")
    print("• 每个码字量化 8 个元素")
    print("• 总共: 1 个 int64 → 32 个原始浮点数")
    print("• abs 部分连续存储在低 32 bits")
    print("• sign 部分交错存储在高 32 bits (SIMD 优化)")
    print("• 详细文档请参考: E8P_QIDXS_BIT_LAYOUT.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
