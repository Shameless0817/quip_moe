CUDA_VISIBLE_DEVICES=0,1,2,3 python quantize_qwen1.5/hessian_offline_qwen1.5_shared_experts.py \
    --base_model Qwen/Qwen1.5-MoE-A2.7B \
    --save_path qwen15_hessian_shared_experts \
    --batch_size 64 \
    --devset_size 1024 \
    --ctx_size 4096 \
    --save_activations