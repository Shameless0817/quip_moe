#!/bin/bash
#SBATCH --job-name=quantize_mixtral
#SBATCH --output=logs/quantize_mixtral_%j.log
#SBATCH --error=logs/quantize_mixtral_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dev
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=512G

# 打印作业信息
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# 切换目录
cd /fact_home/zeyuli/quip_sharp/

# 初始化conda
. /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh

# 激活 conda 环境
conda activate quip_sharp

# 设置CUDA环境变量
export CUDA_HOME=$CONDA_PREFIX
export QUIP_LIB_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib


# 创建日志目录
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /fact_home/zeyuli/quip_sharp/quantize_finetune_mixtral.py \
    --base_model "mistralai/Mixtral-8x7B-v0.1" \
    --save_path ./quantized_mixtral_noft \
    --hessian_path /fact_home/zeyuli/quip_sharp/quantize_mixtral/hessians/mixtral_8x7b \
    --codebook E8P12 \
    --model_type mixtral \
    --batch_size 4 \
    --lora_rank 64 \
    --devset_size 128 \
    --ft_epochs 0 \
    --ft_valid_size 32 \
    --ft_grad_ckpt \
    --scale_override 0.9 
    

echo "End Time: $(date)"