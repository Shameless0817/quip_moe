CUDA_VISIBLE_DEVICES=0,1,2,3 python /fact_home/zeyuli/quip_sharp/quantize_deepseek_vl/hessian_offline_deepseekvl2_experts.py \
    --batch_size 64 \
    --devset_size 1024 \
    --base_model deepseek-ai/deepseek-vl2-small \
    --save_path hessians_deepseekvl2_small_experts \
    --save_activations