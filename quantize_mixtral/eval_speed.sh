# python -m eval.eval_speed  --hf_path /fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_qkvo --seqlen 4096 
# CUDA_VISIBLE_DEVICES=0 python -m eval.eval_speed  --hf_path /fact_data/zeyuli/mixtral_8x7b_quip_full_noft --seqlen 128

# CUDA_VISIBLE_DEVICES=0 python -m eval.eval_speed  --hf_path mistralai/Mixtral-8x7B-v0.1 --seqlen 128 

# CUDA_VISIBLE_DEVICES=0 python -m eval.eval_speed  --hf_path /fact_data/zeyuli/mixtral_8x7b_quip_full_noft --seqlen 4096

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_speed_decode  --hf_path mistralai/Mixtral-8x7B-v0.1 --prefill_seqlen 128 --decode_steps 50 --samples 50 