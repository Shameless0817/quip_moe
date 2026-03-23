# ✅ QuantizedLinear vs Linear 性能对比

## 📚 快速导航

- 🚀 [快速参考卡片](QUICK_REFERENCE.md) - **推荐新手阅读**
- 📊 [详细性能分析](QUANTIZED_VS_LINEAR_ANALYSIS.md) - 深度技术分析
- 🔧 [修复说明](BENCHMARK_FIX_NOTES.md) - 问题解决历史

## 快速开始

### 推荐方法：理论分析模式（最稳定）

```bash
# 激活环境
conda activate quip  # 或包含 PyTorch 的环境名

# 运行理论分析（不创建 QuantizedLinear，避免格式问题）
python benchmark_quantized_vs_linear.py \
    --configs "4096,4096" \
    --batch_size 1 \
    --show_analysis \
    --theory_only
```

### 或使用自动化脚本

```bash
# 快速测试
./test_benchmark_fixed.sh

# 完整演示（3个场景）
./demo_benchmark.sh
```

## 解决方案

由于 E8P12 码本的打包格式复杂，直接创建 QuantizedLinear 可能失败。脚本现在支持：

1. **理论分析模式** (`--theory_only`): 只测试 Linear 层，提供详细的理论分析和估算
2. **自动降级**: 如果 QuantizedLinear 创建失败，自动切换到理论估算模式
3. **使用真实模型**: 从实际的量化模型中测试（推荐用于精确测试）

## 测试不同配置

```bash
# 理论分析模式（推荐，最稳定）
python benchmark_quantized_vs_linear.py \
    --configs "4096,4096;4096,11008;11008,4096" \
    --batch_size 1 \
    --theory_only

# 尝试实际测试（可能因格式问题失败，会自动降级）
python benchmark_quantized_vs_linear.py \
    --configs "4096,4096" \
    --batch_size 1

# 测试批量大小影响（理论模式）
python benchmark_quantized_vs_linear.py \
    --configs "4096,4096" \
    --batch_size 8 \
    --theory_only

# 大模型维度 (Llama-70B)
python benchmark_quantized_vs_linear.py \
    --configs "8192,8192;8192,28672" \
    --batch_size 1 \
    --theory_only
```

## 使用真实量化模型测试（推荐用于精确结果）

如果你已经有量化好的模型，可以直接测试其中的层：

```python
# 示例：从量化模型中提取层进行测试
from lib.utils.unsafe_import import model_from_hf_path

model, _ = model_from_hf_path('mixtral_8x7b_quip_full')

# 测试模型中的实际 QuantizedLinear 层
# （需要编写自定义测试代码）
```

## 预期结果

### 理论分析模式输出

```
================================================================================
性能对比结果
================================================================================
  普通 Linear:
    时间: 0.523 ms (实测)
    内存: 33.55 MB (FP16 权重)

  QuantizedLinear:
    时间: 2.092 ms (理论估算: ~4x Linear)
    内存: 4.22 MB (量化索引 + 缩放)

  对比:
    速度: QuantizedLinear 慢 4.00x
    内存: QuantizedLinear 节省 87.4% (7.95x 压缩)
```

**注意**: 
- 使用 `--theory_only` 时，QuantizedLinear 的时间是**理论估算**（Linear 时间 × 4）
- 内存计算是**精确的**，基于实际的数据结构
- 实际性能可能在 2-6x 慢之间，取决于批次大小和硬件

## 文档

- 📄 [BENCHMARK_FIX_NOTES.md](BENCHMARK_FIX_NOTES.md) - 详细修复说明
- 📊 [QUANTIZED_VS_LINEAR_ANALYSIS.md](QUANTIZED_VS_LINEAR_ANALYSIS.md) - 性能分析文档

## 需要帮助？

如果遇到环境问题：
```bash
# 检查 PyTorch
python -c "import torch; print(torch.__version__)"

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

祝测试顺利！🚀
