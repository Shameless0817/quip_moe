CUDA_VISIBLE_DEVICES=0,1,2,3 python /fact_home/zeyuli/quip_sharp/quantize_finetune_mixtral.py \
    --base_model mistralai/Mixtral-8x7B-v0.1 \
    --save_path mixtral_8x7b_quip_full \
    --hessian_path /fact_home/zeyuli/quip_sharp/mixtral_hessian_qkvo \
    --dense_hessian_path /fact_home/zeyuli/quip_sharp/mixtral_hessian_qkvo \
    --sparse_hessian_path /fact_home/zeyuli/quip_sharp/mixtral_hessian_experts \
    --codebook E8P12 \
    --model_type mixtral \
    --batch_size 32 \
    --lora_rank 0 \
    --devset_size 128 \
    --ft_epochs 0 \
    --ft_valid_size 32 \
    --ft_grad_ckpt \
    --scale_override 0.9 
    



# python -m /fact_home/zeyuli/quip_sharp/quantize_finetune_mixtral.py 
#     --save_path $CKPT/2_7b_2bit 
#     --codebook E8P12  
#     --scale_override 0.9 
#     --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144/ --devset_size 384 --ft_valid_size 128