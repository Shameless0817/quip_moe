# Qwen1.5-MoE-A2.7B
CUDA_VISIBLE_DEVICES=0,1,2,3 python quantize_qwen1.5/hessian_offline_qwen1.5_experts.py \
    --base_model Qwen/Qwen1.5-MoE-A2.7B \
    --save_path qwen15_hessian_experts \
    --batch_size 16 \
    --devset_size 1024 \
    --ctx_size 4096 \
    --save_activations \
    --moe_only
  
# python quantize_mixtral/hessian_offline_mixtral_experts.py \
#     --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --save_path mixtral_hessian_experts \
#     --batch_size 16 \
#     --devset_size 1024 \
#     --save_activations