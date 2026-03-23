#!/bin/bash
# 性能对比测试 - 一键启动

clear
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║    QuantizedLinear vs Linear 性能对比工具                    ║"
echo "║                                                               ║"
echo "║    快速、简单、可靠的性能分析                                ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 检查环境
if ! python -c "import torch" 2>/dev/null; then
    echo "⚠️  请先激活包含 PyTorch 的环境:"
    echo ""
    echo "    conda activate quip  # 或你的环境名"
    echo ""
    exit 1
fi

echo "✓ 环境检查通过"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  选择你的操作:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1) 快速测试 (推荐新手)"
echo "     → 单个配置 + 理论分析"
echo ""
echo "  2) 完整演示"
echo "     → 3个场景 + 保存结果"
echo ""
echo "  3) 查看文档"
echo "     → 快速参考 + 使用指南"
echo ""
echo "  4) 自定义测试"
echo "     → 手动指定参数"
echo ""
echo "  5) 退出"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "请选择 [1-5]: " choice
echo ""

case $choice in
    1)
        echo "▶ 运行快速测试..."
        echo ""
        ./test_benchmark_fixed.sh
        ;;
    2)
        echo "▶ 运行完整演示..."
        echo ""
        ./demo_benchmark.sh
        ;;
    3)
        echo "▶ 文档列表:"
        echo ""
        echo "  - BENCHMARK_DOCS_INDEX.md    # 文档索引"
        echo "  - QUICK_REFERENCE.md         # 快速参考 ⭐"
        echo "  - README_BENCHMARK.md        # 使用指南"
        echo "  - COMPLETE_FIX_SUMMARY.md    # 完整总结"
        echo ""
        echo "推荐首读: QUICK_REFERENCE.md"
        echo ""
        read -p "按回车查看快速参考..." 
        cat QUICK_REFERENCE.md | less
        ;;
    4)
        echo "▶ 自定义测试"
        echo ""
        echo "示例命令:"
        echo ""
        echo "  python benchmark_quantized_vs_linear.py \\"
        echo "      --configs \"4096,4096;4096,11008\" \\"
        echo "      --batch_size 1 \\"
        echo "      --show_analysis \\"
        echo "      --theory_only"
        echo ""
        echo "查看所有选项:"
        echo "  python benchmark_quantized_vs_linear.py --help"
        echo ""
        ;;
    5)
        echo "再见！"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  测试完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📚 了解更多:"
echo "    - 查看文档: cat QUICK_REFERENCE.md"
echo "    - 文档索引: cat BENCHMARK_DOCS_INDEX.md"
echo "    - 再次运行: ./start_benchmark.sh"
echo ""
