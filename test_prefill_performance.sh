#!/bin/bash
# 测试不同 prefill size 下的性能

# 使用示例 1: 测试量化模型
python eval/eval_prefill_decode.py \
    --hf_path mixtral_8x7b_quip_full \
    --batch_size 1 \
    --prefill_sizes "32,64,128,256,512,1024,2048,4096" \
    --decode_steps 128 \
    --warmup_samples 3 \
    --test_samples 10 \
    --output_json prefill_results_mixtral_quip.json

# 使用示例 2: 测试原始模型（如果需要对比）
# python eval/eval_prefill_decode.py \
#     --hf_path meta-llama/Llama-2-7b-hf \
#     --batch_size 1 \
#     --prefill_sizes "32,64,128,256,512,1024,2048" \
#     --decode_steps 128 \
#     --warmup_samples 3 \
#     --test_samples 10 \
#     --output_json prefill_results_llama2_7b.json

# 使用示例 3: 快速测试（更少的 prefill sizes 和样本数）
# python eval/eval_prefill_decode.py \
#     --hf_path mixtral_8x7b_quip_qkvo \
#     --batch_size 1 \
#     --prefill_sizes "128,512,2048" \
#     --decode_steps 64 \
#     --warmup_samples 2 \
#     --test_samples 5 \
#     --output_json quick_test.json
