#!/bin/bash
#SBATCH --job-name=test_compare
#SBATCH --time=00:30:00
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --output=/fact_home/zeyuli/quip_sharp/test_compare.log
#SBATCH --error=/fact_home/zeyuli/quip_sharp/test_compare.err

cd /fact_home/zeyuli/quip_sharp

# 运行脚本，仅比较1层以加快测试
python compare_mem_efficient.py --num_layers 1 --text "The capital of France is"
