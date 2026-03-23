#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3 python /fact_home/zeyuli/quip_sharp/quantize_deepseek/hessian_offline_deepseek_qkvo.py \
    --batch_size 32 \
    --devset_size 1024 \
    --ctx_size 4096 \
    --base_model deepseek-ai/deepseek-moe-16b-base \
    --save_path hessians_deepseek_moe_16b_base_qkvo \
    --save_activations