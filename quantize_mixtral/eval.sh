# CUDA_VISIBLE_DEVICES=0 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 & 


CUDA_VISIBLE_DEVICES=3 python -m eval.eval_ppl  --hf_path /fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_o_only --seqlen 4096 