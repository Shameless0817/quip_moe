#!/bin/bash

# DEEPSEEK 模型对比脚本使用说明
# 用于诊断权重迁移中的问题

echo "========================================"
echo "DeepSeek Model Comparison - Debug Guide"
echo "========================================"
echo ""

# 检查GPU
echo "[1] Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# 清理之前的缓存
echo "[2] Clearing cache..."
python3 -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
echo ""

# 运行简化版对比脚本
echo "[3] Running simplified comparison script..."
echo "    (This script uses device_map='auto' to distribute model across GPUs)"
echo ""

python3 compare_simple.py \
    --model_name "deepseek-ai/deepseek-moe-16b-base" \
    --num_layers 5 \
    --text "The capital of France is"

echo ""
echo "========================================"
echo "Comparison completed!"
echo "========================================"
