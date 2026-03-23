# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq --batch_size 32 --hf_path  deepseek-ai/deepseek-moe-16b-base
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq --batch_size 32 --hf_path /fact_home/zeyuli/quip_sharp/deepseek_quip_full_noft
# CUDA_VISIBLE_DEVICES=3,2,1,0 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy --batch_size 64 --hf_path /fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_full_noft

# CUDA_VISIBLE_DEVICES=3,2,1,0 python -m eval.eval_zeroshot --tasks arc_challenge --batch_size 128 --hf_path /fact_data/zeyuli/mixtral_8x7b_quip_full_noft
# CUDA_VISIBLE_DEVICES=3,2,1,0 python -m eval.eval_zeroshot --tasks arc_easy --batch_size 128 --hf_path /fact_data/zeyuli/mixtral_8x7b_quip_full_noft