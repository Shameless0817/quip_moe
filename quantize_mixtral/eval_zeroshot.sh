# CUDA_VISIBLE_DEVICES=2 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 32 --hf_path /fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_qkvo 
# CUDA_VISIBLE_DEVICES=3,2,1,0 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy --batch_size 64 --hf_path /fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_full_noft

# CUDA_VISIBLE_DEVICES=3,2,1,0 python -m eval.eval_zeroshot --tasks arc_challenge --batch_size 128 --hf_path /fact_data/zeyuli/mixtral_8x7b_quip_full_noft
CUDA_VISIBLE_DEVICES=3,2,1,0 python -m eval.eval_zeroshot --tasks arc_easy --batch_size 128 --hf_path /fact_data/zeyuli/mixtral_8x7b_quip_full_noft