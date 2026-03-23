#!/bin/bash
# 演示：QuantizedLinear vs Linear 性能对比（理论模式）

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   QuantizedLinear vs Linear 性能对比工具                  ║"
echo "║   理论分析模式 - 最稳定可靠                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查环境
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ 错误: 找不到 PyTorch"
    echo ""
    echo "请先激活正确的环境:"
    echo "  conda activate quip  # 或你的环境名"
    echo ""
    exit 1
fi

echo "✓ PyTorch 环境已就绪"
echo ""

# 演示 1: 单个配置 + 详细分析
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "演示 1: 单个配置 (4096×4096) + 理论分析"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python benchmark_quantized_vs_linear.py \
    --configs "4096,4096" \
    --batch_size 1 \
    --warmup 3 \
    --iterations 20 \
    --show_analysis \
    --theory_only

echo ""
echo ""

# 演示 2: 多个配置 (Llama-7B 维度)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "演示 2: Llama-7B 典型层维度"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python benchmark_quantized_vs_linear.py \
    --configs "4096,4096;4096,11008;11008,4096" \
    --batch_size 1 \
    --warmup 3 \
    --iterations 15 \
    --theory_only \
    --output_json demo_llama7b_results.json

echo ""
echo ""

# 演示 3: 批量大小影响
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "演示 3: 批量大小影响 (batch_size=8)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python benchmark_quantized_vs_linear.py \
    --configs "4096,4096" \
    --batch_size 8 \
    --warmup 3 \
    --iterations 10 \
    --theory_only

echo ""
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   演示完成！                                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 结果已保存: demo_llama7b_results.json"
echo ""
echo "📖 文档:"
echo "  - 快速参考: cat QUICK_REFERENCE.md"
echo "  - 详细分析: cat QUANTIZED_VS_LINEAR_ANALYSIS.md"
echo "  - 修复说明: cat BENCHMARK_FIX_NOTES.md"
echo ""
echo "🎯 关键结论:"
echo "  ✓ QuantizedLinear ~4x 慢于 Linear"
echo "  ✓ 但节省 ~87% 内存 (8x 压缩)"
echo "  ✓ 批量越大，性能差距越小"
echo ""
