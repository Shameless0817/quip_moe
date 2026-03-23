# E8P12 Qidxs Bit 排列详解

## 概述

E8P12 (E8 Padded 12-bit) 是一个 2-bit 量化方案，每个码字 (codebook entry) 由 **16 bits** 表示，用于量化 **8 个浮点数**。

## 基本参数

- **codesz** = 8：每个码字量化 8 个元素
- **packsz** = 4：打包因子
- **每个索引**: 16 bits (存储为 int64 的低 16 位)
- **量化后形状**: `[m, n // 32]`，其中 m 是输出维度，n 是输入维度

## 量化流程

### 1. 量化阶段 (LDLQ 算法)

```python
# 在 lib/algo/quip.py 中
Qidxs = torch.zeros(m, n // cb.codesz, dtype=torch.int64)  # [m, n // 8]

# 对每个位置，量化 8 个元素得到一个 16-bit 索引
for k in range(n // 8):
    hatWr[:, k*8:(k+1)*8], Qidxs[:, k] = cb.quantize(WXWX)
```

**输出**: `Qidxs` 形状为 `[m, n // 8]`，每个元素是 16-bit int64

### 2. 打包阶段 (maybe_pack_idxs)

打包从 `[m, n // 8]` → `[m, n // 32]`，将 4 个 16-bit 索引打包成 1 个 64-bit int64。

```python
# 在 lib/codebook/latticee8_padded12.py::maybe_pack_idxs
def maybe_pack_idxs(self, idxs):
    m, n = idxs.shape  # [m, n//8]
    
    # 第一步：重排为 [m//2, n//16, 2, 2]
    idxs = idxs.view(m // 2, 2, (n * 8) // 16, 2).transpose(1, 2)
    # 现在是 [m//2, n//16, 2, 2]
    
    # 提取 abs 部分（高 8 bits）
    abs32 = (idxs[:, :, 0, 0] >> 8) + \
            ((idxs[:, :, 1, 0] >> 8) << 8) + \
            ((idxs[:, :, 0, 1] >> 8) << 16) + \
            ((idxs[:, :, 1, 1] >> 8) << 24)
    
    # 提取 sign 部分（低 8 bits）
    sign32 = torch.zeros_like(abs32)
    for i in range(4):  # 4 个码字
        wt = idxs[:, :, i % 2, i // 2]
        for j in range(8):  # 每个码字的 8 个 sign bits
            sign32 += ((wt >> j) & 1) << (4 * j + i)
    
    # 组合成 64-bit
    output = (sign32 << 32) + abs32
    
    # 最终 reshape
    output = output.reshape(m // 16, 8, n // 8, 4).transpose(1, 2)
    return output.view(m, n // 4)  # [m, n//32]
```

## 每个 int64 的 Bit 布局

每个 64-bit int64 编码了 **4 个 16-bit 码字索引**，每个码字量化 **8 个元素**，所以总共量化 **32 个元素**。

### 64-bit 结构:

```
Bits [63:32] - sign32 (32 bits of sign information)
Bits [31:0]  - abs32  (32 bits of absolute value indices)
```

### abs32 详细结构 (Bits [31:0]):

```
Bits [31:24] - abs_idx[3]  (第 4 个码字的 abs 部分)
Bits [23:16] - abs_idx[2]  (第 3 个码字的 abs 部分)
Bits [15:8]  - abs_idx[1]  (第 2 个码字的 abs 部分)
Bits [7:0]   - abs_idx[0]  (第 1 个码字的 abs 部分)
```

每个 `abs_idx[i]` 是一个 **8-bit** 索引，指向 256 个 codebook 条目之一。

### sign32 详细结构 (Bits [63:32]):

Sign bits 采用交错排列，用于 SIMD 友好的解码：

```
对于每个码字 i (0-3)，它的 8 个 sign bits 分布在:
  码字 0: bits [32, 36, 40, 44, 48, 52, 56, 60] (每隔 4 bits)
  码字 1: bits [33, 37, 41, 45, 49, 53, 57, 61]
  码字 2: bits [34, 38, 42, 46, 50, 54, 58, 62]
  码字 3: bits [35, 39, 43, 47, 51, 55, 59, 63]
```

具体打包公式（在 maybe_pack_idxs 中）:
```python
for i in range(4):  # 4 个码字
    wt = idxs[:, :, i % 2, i // 2]  # 获取第 i 个码字
    for j in range(8):  # 8 个元素的 sign bits
        sign32 += ((wt >> j) & 1) << (4 * j + i)
        # bit j 的 sign 被放在位置 4*j + i
```

## 解压缩阶段

### CUDA Kernel 中的解压逻辑

在 `decompress_packed_e8p_kernel` 中：

```cuda
uint2 w_compr = weights_compressed[idx];  // 读取 64-bit
uint32_t a = w_compr.x;  // abs32 (低 32 bits)
uint32_t b = w_compr.y;  // sign32 (高 32 bits)

// 处理每个码字
for (int block = 0; block < 4; block++) {
    // 提取 abs 索引
    uint32_t abs_idx = (a >> (block * 8)) & 255;
    
    // 从 codebook 获取绝对值向量 (8 个 half)
    uint32_t x = codebook_abs[abs_idx];
    
    // 提取并应用 sign bits
    uint32_t signs = (b >> block) & 0x11111111;  // 每隔 4 bits 取一个
    x = x ^ (signs * 14);  // 应用符号翻转
    
    // 解码为 8 个 half precision 值
    // ... (详细的 bit 操作)
}
```

### Reshape 要求

CUDA kernel 期望输入形状为 `[m//16, n//64, 8, 4]`:

- **第一维** `m//16`: 每 16 行为一组（CUDA block 处理单位）
- **第二维** `n//64`: 每 64 列为一组（一个 warp 处理 2 个 32 元素块）
- **第三维** `8`: 每组 8 个 int64
- **第四维** `4`: 每个 int64 包含 4 个码字

**验证**: 
- 总元素数 = `(m//16) * (n//64) * 8 * 4 = m * n / 32` ✓
- 每个元素编码 32 个输入 → 总输入 = `m * n / 32 * 32 = m * n` ✓

## 示例

假设 `m=4096, n=4096`:

1. **量化后**: `Qidxs = [4096, 512]` (每个元素 16 bits)
2. **打包后**: `Qidxs = [4096, 128]` (每个元素 64 bits, int64)
3. **Reshape**: `Qidxs.view(256, 64, 8, 4)` 用于 CUDA kernel

每个 `Qidxs[i, j]` 的 64 bits 编码:
- 4 个码字索引 (每个 16 bits)
- 每个码字量化 8 个元素
- 总共量化 32 个原始权重值

## 为什么这样设计？

1. **内存效率**: 2 bits/element 的平均压缩率
2. **SIMD 友好**: Sign bits 的交错排列允许并行处理
3. **Cache 友好**: 16 行 × 64 列的块对齐
4. **Warp 效率**: 每个 warp (32 threads) 可以并行处理一组数据

## 关键函数总结

| 函数 | 输入形状 | 输出形状 | 作用 |
|------|---------|---------|------|
| `quantize` | `[m, n]` float | `[m, n//8]` int64 | 量化，每 8 个元素→1 个 16-bit 索引 |
| `maybe_pack_idxs` | `[m, n//8]` int64 | `[m, n//32]` int64 | 打包，4 个 16-bit → 1 个 64-bit |
| `decompress_packed_e8p` | `[m//16, n//64, 8, 4]` int64 | `[m, n]` float16 | 解压缩为原始权重 |

## 参考代码位置

- 量化: `lib/algo/quip.py::LDLQ`
- 打包: `lib/codebook/latticee8_padded12.py::maybe_pack_idxs`
- 解压: `quiptools/quiptools_e8p_gemv.cu::decompress_packed_e8p_kernel`
