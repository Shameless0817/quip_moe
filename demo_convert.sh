CUDA_VISIBLE_DEVICES=1,2,3 python /fact_home/zeyuli/quip_sharp/convert_lora_mixtral.py \
    --quantized_model /fact_data/zeyuli/mixtral_8x7b_quip_full_noft \
    --original_model mistralai/Mixtral-8x7B-v0.1  \
    --output_dir mixtral_lora_converted_hf_v2 \
    --lora_rank 16 \
    --num_layers 32