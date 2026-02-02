#!/bin/bash
#SBATCH --job-name=hessian_mixtral
#SBATCH --output=logs/hessian_mixtral_%j.log
#SBATCH --error=logs/hessian_mixtral_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ai
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=220:00:00

# 初始化shell环境
set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# 初始化conda
. /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh

# 激活conda环境
conda activate quip_sharp

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

# 设置环保模式参数，减少内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 运行任务，减少batch size和数据集大小以节省内存
python quantize_mixtral/hessian_offline_mixtral.py \
    --base_model mistralai/Mixtral-8x7B-v0.1 \
    --save_path mixtral_hessian \
    --batch_size 4 \
    --devset_size 64 \
    --ctx_size 2048 \
    --save_activations \

echo "End Time: $(date)"
echo "Job Status: $?"