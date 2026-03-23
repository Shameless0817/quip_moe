# python quantize_mixtral/hessian_offline_mixtral_experts.py \
#     --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --save_path mixtral_hessian_experts \
#     --batch_size 16 \
#     --devset_size 1024 \
#     --save_activations


python quantize_qwen2/hessian_offline_qwen2_experts.py \
    --base_model Qwen/Qwen2-57B-A14B-Instruct \
    --save_path ./hessians_qwen2moe \
    --batch_size 4 \
    --devset_size 512 \
    --ctx_size 4096 \
    --save_activations \
    --moe_only