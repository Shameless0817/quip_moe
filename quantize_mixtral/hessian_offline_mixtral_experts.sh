#!/bin/bash

# echo "Start Time: $(date)"

# python quantize_mixtral/hessian_offline_mixtral_experts.py \
#     --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --save_path mixtral_hessian_experts \
#     --batch_size 8 \
#     --devset_size 1024 \
#     --ctx_size 4096 \
#     --save_activations

# echo "End Time: $(date)"


echo "Start Time: $(date)"

python quantize_mixtral/hessian_offline_mixtral_experts.py \
    --base_model mistralai/Mixtral-8x22B-v0.1 \
    --save_path /fact_data/zeyuli/mixtral_22B_hessian_experts \
    --batch_size 8 \
    --devset_size 1024 \
    --ctx_size 4096 \
    --save_activations

echo "End Time: $(date)"
