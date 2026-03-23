python /fact_home/zeyuli/quip_sharp/quantize_qwen3_vl/hessian_offline_qwen3_qkvo.py \
    --base_model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --save_path ./hessians_qwen3_vl_qkvo \
    --batch_size 16 \
    --devset_size 1024 \
    --save_activations