# Prefill & Decode 性能测试工具

## 功能说明

`eval_prefill_decode.py` 是一个用于测试大语言模型在不同 prefill size 下性能表现的工具。它会测量：

### 测试指标

1. **Prefill 阶段**（第一次前向传播，处理完整上下文）
   - 延迟 (Latency): 毫秒 (ms)
   - 吞吐量 (Throughput): tokens/秒

2. **Decode 阶段**（使用 KV cache 的增量生成）
   - 单 token 延迟: 毫秒/token (ms/token)
   - 吞吐量: tokens/秒

3. **完整生成**（Prefill + N 步 Decode）
   - 总时间: 秒 (s)
   - 平均吞吐量: tokens/秒

## 使用方法

### 基本用法

```bash
python eval/eval_prefill_decode.py \
    --hf_path <模型路径> \
    --prefill_sizes "32,64,128,256,512,1024,2048" \
    --decode_steps 128 \
    --output_json results.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hf_path` | meta-llama/Llama-2-7b-hf | 模型路径 |
| `--batch_size` | 1 | 批次大小 |
| `--prefill_sizes` | 32,64,128,256,512,1024,2048 | 要测试的 prefill size 列表（逗号分隔）|
| `--decode_steps` | 128 | Decode 阶段生成的 token 数量 |
| `--warmup_samples` | 5 | 预热迭代次数 |
| `--test_samples` | 20 | 每个 prefill size 的测试迭代次数 |
| `--no_use_cuda_graph` | False | 禁用 CUDA graph |
| `--no_use_flash_attn` | False | 禁用 Flash Attention |
| `--output_json` | None | 保存 JSON 结果的路径 |

### 使用示例脚本

```bash
# 方式 1: 直接运行测试脚本
./test_prefill_performance.sh

# 方式 2: 自定义参数
python eval/eval_prefill_decode.py \
    --hf_path mixtral_8x7b_quip_full \
    --batch_size 1 \
    --prefill_sizes "128,512,1024,2048" \
    --decode_steps 64 \
    --warmup_samples 3 \
    --test_samples 10 \
    --output_json my_results.json
```

## 输出示例

### 终端输出

```
==========================================
Testing prefill size: 512
==========================================
Benchmarking prefill phase...
Benchmarking decode phase...
Benchmarking full generation (128 decode steps)...

Results for prefill_size=512:
  Prefill:
    Latency: 45.23 ms
    Throughput: 11324.50 tokens/s
  Decode:
    Latency: 12.34 ms/token
    Throughput: 81.04 tokens/s
  Full Generation (prefill + 128 decode):
    Total time: 1.62 s
    Average throughput: 395.06 tokens/s
```

### 汇总表格

```
====================================================================================================
SUMMARY TABLE
====================================================================================================
Prefill Size    Prefill (ms)    Prefill (tok/s)    Decode (ms)     Decode (tok/s)     Full Gen (tok/s)  
----------------------------------------------------------------------------------------------------
32              5.23            6118.18            12.45           80.32              85.23             
64              8.45            7573.96            12.38           80.77              120.45            
128             15.67           8168.47            12.41           80.58              190.34            
256             28.90           8858.48            12.35           80.97              295.67            
512             45.23           11324.50           12.34           81.04              395.06            
1024            82.14           12467.89           12.30           81.30              605.34            
2048            158.90          12889.76           12.28           81.43              890.12            
====================================================================================================
```

### JSON 输出

```json
{
  "model": "mixtral_8x7b_quip_full",
  "batch_size": 1,
  "decode_steps": 128,
  "use_cuda_graph": true,
  "use_flash_attn": true,
  "results": [
    {
      "prefill_size": 512,
      "batch_size": 1,
      "decode_steps": 128,
      "prefill": {
        "latency_ms": 45.23,
        "tokens_per_sec": 11324.50
      },
      "decode": {
        "latency_ms": 12.34,
        "tokens_per_sec": 81.04
      },
      "full_generation": {
        "total_time_s": 1.62,
        "total_tokens": 640,
        "tokens_per_sec": 395.06
      }
    }
  ]
}
```

## 指标解读

### Prefill vs Decode 性能差异

- **Prefill 阶段**: 
  - 计算密集型（compute-bound）
  - 可以充分利用 GPU 并行计算
  - 吞吐量通常较高（数千到上万 tokens/s）
  - 延迟随 prefill size 增长

- **Decode 阶段**:
  - 内存访问密集型（memory-bound）
  - 每次只生成一个 token，需要加载整个 KV cache
  - 吞吐量通常较低（几十到几百 tokens/s）
  - 延迟相对稳定，不随 prefill size 显著变化

### 性能优化关注点

1. **Prefill 优化**: 提高吞吐量
   - Flash Attention
   - 更高效的矩阵乘法
   - 量化减少计算量

2. **Decode 优化**: 降低延迟
   - KV cache 优化
   - 更快的内存访问
   - CUDA graph 减少启动开销

## 实际应用场景

- **长上下文应用**: 关注 prefill 性能（如文档问答、代码补全）
- **对话系统**: 关注 decode 性能（低延迟单 token 生成）
- **批量推理**: 关注整体吞吐量

## 注意事项

1. 首次运行会有模型加载时间，不影响测量结果
2. 建议在独占 GPU 环境下测试以获得稳定结果
3. `--warmup_samples` 确保 CUDA kernel 充分预热
4. `--test_samples` 越多结果越稳定，但测试时间越长
5. 不同模型架构的性能特征可能差异很大
