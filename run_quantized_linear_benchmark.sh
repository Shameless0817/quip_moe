#!/bin/bash
# QuantizedLinear vs Linear 性能对比测试

echo "测试 1: 常见的 LLM 层维度 (Llama-7B)"
python benchmark_quantized_vs_linear.py \
    --configs "4096,4096;4096,11008;11008,4096" \
    --batch_size 1 \
    --warmup 10 \
    --iterations 100 \
    --show_analysis \
    --output_json results_llama7b_dims.json

# echo -e "\n\n测试 2: 不同批次大小的影响"
# python benchmark_quantized_vs_linear.py \
#     --configs "4096,4096" \
#     --batch_size 8 \
#     --warmup 10 \
#     --iterations 50 \
#     --output_json results_batch8.json

# echo -e "\n\n测试 3: 更大的模型维度 (Llama-13B/70B)"
# python benchmark_quantized_vs_linear.py \
#     --configs "5120,5120;5120,13824;8192,8192" \
#     --batch_size 1 \
#     --warmup 10 \
#     --iterations 100 \
#     --show_analysis \
#     --output_json results_large_models.json

echo -e "\n\n测试完成！"
