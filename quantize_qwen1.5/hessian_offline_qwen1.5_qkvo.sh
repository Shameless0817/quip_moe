python quantize_qwen1.5/hessian_offline_qwen1.5_qkvo.py \
    --base_model Qwen/Qwen1.5-MoE-A2.7B \
    --save_path ./qwen15_hessian_qkvo \
    --batch_size 16 \
    --devset_size 1024 \
    --ctx_size 4096 \
    --save_activations