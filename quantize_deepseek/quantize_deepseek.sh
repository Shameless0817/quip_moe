#!/bin/bash
set -euo pipefail

ROOT_DIR="/fact_home/zeyuli/quip_sharp"
LOG_DIR="${ROOT_DIR}/quantize_deepseek/logs"
SAVE_PATH="${ROOT_DIR}/deepseek_quip_full"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/experts_proxy_$(date +%Y%m%d_%H%M%S).log"

cd "${ROOT_DIR}"

# 注意：
# - 如果 ft_epochs=0 (不进行微调)，则 ft_bs 可以设置为任意值  
# - 如果 ft_epochs>0 (进行微调)，请设置合理的 ft_bs 值(如 4 或 8)
# - ft_valid_size 必须小于 devset_size，建议设置为 devset_size 的 1/4 到 1/8

# q k v o量化
# python /fact_home/zeyuli/quip_sharp/quantize_deepseek/quantize_finetune_deepseek_s.py \
#     --base_model deepseek-ai/deepseek-moe-16b-base \
#     --model_type deepseek \
#     --save_path deepseek_quip_qkvo \
#     --codebook E8P12 \
#     --dense_hessian_path /fact_home/zeyuli/quip_sharp/hessians_deepseek_moe_16b_base_qkvo \
#     --sparse_hessian_path /fact_home/zeyuli/quip_sharp/hessians_deepseek_moe_16b_base_experts \
#     --devset_size 128 \
#     --batch_size 32 \
#     --ft_epochs 0 \
#     --ft_valid_size 32 \
#     --run_generation_test


# quantized all
python /fact_home/zeyuli/quip_sharp/quantize_deepseek/quantize_finetune_deepseek_s.py \
    --base_model deepseek-ai/deepseek-moe-16b-base \
    --model_type deepseek \
    --save_path "${SAVE_PATH}" \
    --codebook E8P12 \
    --dense_hessian_path /fact_home/zeyuli/quip_sharp/hessians_deepseek_moe_16b_base_qkvo \
    --sparse_hessian_path /fact_home/zeyuli/quip_sharp/hessians_deepseek_moe_16b_base_experts \
    --devset_size 128 \
    --batch_size 32 \
    --ft_epochs 0 \
    --ft_valid_size 32 \
    --run_generation_test 2>&1 | tee "${LOG_FILE}"

echo "Quantized weights saved to: ${SAVE_PATH}"
echo "Quantization log saved to: ${LOG_FILE}"