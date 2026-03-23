# CUDA_VISIBLE_DEVICES=0 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 & 


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.eval_ppl  --hf_path mistralai/Mixtral-8x7B-v0.1 --seqlen 4096 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.eval_lora_ppl  --hf_path /fact_home/zeyuli/quip_sharp/mixtral_lora_converted_hf --seqlen 4096 --use_lora