#!/bin/bash
#SBATCH --job-name=quantize_llama
#SBATCH --output=logs/quantize_llama_%j.log
#SBATCH --error=logs/quantize_llama_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ai
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=1188:00:00

# 打印作业信息
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# 切换目录
cd /fact_home/zeyuli/quip_sharp/quantize_mixtral

# 初始化conda（sbatch环境需要显式初始化）
conda init
source activate quip_sharp

# 设置CUDA环境变量
export CUDA_HOME=$CONDA_PREFIX
export QUIP_LIB_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib

# 显示环境信息
echo "========================================="
echo "Environment Information:"
echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "========================================="

# 创建日志目录
mkdir -p logs

# 运行任务
@REM python hessian_offline_mixtral.py \
@REM     --base_model mistralai/Mixtral-8x7B-v0.1 \
@REM     --save_path hessians/mixtral_8x7b \
@REM     --batch_size 2 \
@REM     --devset_size 256 \
@REM     --ctx_size 4096 \
@REM     --save_activations
. ./example.sh

echo "End Time: $(date)"