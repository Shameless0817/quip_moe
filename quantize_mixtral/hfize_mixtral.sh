# Example script for quantizing mixtral 2 7b with QuIP#

# CKPT=ckpt
# HF=hf
# LOG=log
# HESS=hess

# mkdir $CKPT
# mkdir $HF
# mkdir $LOG

# python /fact_home/zeyuli/quip_sharp/quantize_mixtral/hfize_mixtral.py \
#     --quantized_path ./quantized_mixtral_noft_v2 \
#     --hf_output_path ./mixtral_8x7b_quip_v2

# CUDA_VISIBLE_DEVICES=0 python /fact_home/zeyuli/quip_sharp/quantize_mixtral/hfize_mixtral.py \
#     --quantized_path /fact_home/zeyuli/quip_sharp/quantized_mixtral_o_only \
#     --hf_output_path ./mixtral_8x7b_quip_o_only


# CUDA_VISIBLE_DEVICES=1 python /fact_home/zeyuli/quip_sharp/quantize_mixtral/hfize_mixtral.py \
#     --quantized_path /fact_home/zeyuli/quip_sharp/quantized_mixtral_qkvo \
#     --hf_output_path ./mixtral_8x7b_quip_qkvo


CUDA_VISIBLE_DEVICES=0,1 python /fact_home/zeyuli/quip_sharp/quantize_mixtral/hfize_mixtral.py \
    --quantized_path /fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_full \
    --hf_output_path ./mixtral_8x7b_quip_full_noft