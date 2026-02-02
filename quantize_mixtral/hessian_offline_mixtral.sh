python quantize_mixtral/hessian_offline_mixtral_original.py \
    --base_model mistralai/Mixtral-8x7B-v0.1 \
    --save_path mixtral_hessian_new_v2 \
    --batch_size 32 \
    --devset_size 256 \
    --ctx_size 4096 \
    --save_activations


echo "End Time: $(date)"