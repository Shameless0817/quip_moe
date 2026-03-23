- 原理
![alt text](image.png)
右乘hadamard矩阵平滑权重空间
左乘hadamard矩阵平滑输入空间

- 命令
生成Hessian矩阵命令：
1. 只生成experts的Hessian矩阵
python quantize_mixtral/hessian_offline_mixtral_experts.py \
    --base_model mistralai/Mixtral-8x7B-v0.1 \
    --save_path mixtral_hessian_experts \
    --batch_size 16 \
    --devset_size 1024 \
    --save_activations
e.g.: batch size 16刚好能装下一个46GB的卡, 非常慢



2. 转成huggingface格式存储

python quantize_mixtral/hfize_mixtral.py \
    --quantized_path ./quantized_mixtral_qkvo \
    --hf_output_path ./mixtral_8x7b_quip_qkvo

CUDA_VISIBLE_DEVICES=1 python /fact_home/zeyuli/quip_sharp/quantize_mixtral/hfize_mixtral.py \ 
    --quantized_path /fact_home/zeyuli/quip_sharp/quantized_mixtral_qkvo \
    --hf_output_path ./mixtral_8x7b_quip_qkvo


CUDA_VISIBLE_DEVICES=0,1 python /fact_home/zeyuli/quip_sharp/quantize_mixtral/hfize_mixtral.py \
    --quantized_path /fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_full \
    --hf_output_path ./mixtral_8x7b_quip_full_noft



3. 量化脚本
# 运行量化（确保已经收集了 Q/K/V/O 的 Hessian 矩阵）

CUDA_VISIBLE_DEVICES=0,1 python /fact_home/zeyuli/quip_sharp/quantize_finetune_mixtral.py \
    --base_model mistralai/Mixtral-8x7B-v0.1 \
    --save_path ./quantized_mixtral_qkvo \
    --hessian_path ./mixtral_hessian_qkvo \
    --dense_hessian_path /fact_home/zeyuli/quip_sharp/mixtral_hessian_qkvo \
    --sparse_hessian_path /fact_home/zeyuli/quip_sharp/mixtral_hessian_experts \
    --codebook E8P12 \
    --use_fp64 \
    --rescale_WH \
    --devset_size 384 \
    --batch_size 8
    --ft_valid_size 128 \
    --ft_epochs 3 \
    --ft_bs 1 \
    --ctx_size 4096 \
    --ft_update_freeq 2 \
    --ft_train_mode \
    --ckpt_path mixtral_qkvo_f3/2_7b_2bit


- 结果
在PPT里面


- 现在加上Residual Compensator

4. 测试两个linear层的速度差异
python benchmark_quantized_vs_linear.py --batch_size 1

====================================================================================================
维度                   Linear (ms)     Quantized (ms)  速度比          内存比         
----------------------------------------------------------------------------------------------------
4096x4096            0.021           164.315         7693.53x (快) 0.13        
4096x11008           0.244           84.762          347.08x (快)  0.13        
11008x4096           0.243           84.110          346.80x (快)  0.13        
====================================================================================================

可以看到，quip乘法相比较来说还是挺慢的

5. 加上lora补偿的方法 
. ./demo_convert.sh

python /fact_home/zeyuli/quip_sharp/convert_lora_mixtral.py \
    --quantized_model /fact_data/zeyuli/mixtral_8x7b_quip_full_noft \
    --original_model mistralai/Mixtral-8x7B-v0.1  \
    --output_dir mixtral_lora_converted_hf \
    --lora_rank 16 \
    --num_layers 32

Question:现在实验扩展到qwen上有个问题是qwen架构的维度不是hadamard可以得到矩阵的维度
采用kronech内积的形式代替hadamard，暂时搁置这块的实现


准备先解决速度问题：
对于batch_size > 1的case，我直接将2bit权重完全解压缩成FP16格式并写入HBM，矩阵非常大，紧接着又用矩阵乘法中将这些数据从HBM一步一步读回矩阵单元

代码screenshot
W_decompressed = torch.ops.quip_lib.decompress_packed_e8p(
    Qidxs_list[0].view(m // 16, n // 64, 8, 4),
    self.codebook.grid_packed_abs,
    m, n)
x = (x.to(torch.float16) @ W_decompressed.T).to(torch.float32)

batch_size > 1的情况，思路Fused Dequantize + GEMM Kernel来解决

batch size < 1的情况，代码路径如下：

# 原始流程
x = x * SU                                      # Kernel 1: Element-wise mul
x = matmul_hadU_cuda(x, had_left_T, K_left)     # Kernel 2: Hadamard
x = x / self.scale                              # Kernel 3: Element-wise div
# --- 核心瓶颈 ---
x = torch.ops.quip_lib.decode_matvec_e8p(...)   # Kernel 4: Custom CUDA GEMV
# ----------------
x = matmul_hadU_cuda(x, had_right, K_right)     # Kernel 5: Hadamard
x = x * SV * self.scale                         # Kernel 6: Element-wise mul

思路：
写一个triton kernel融合 Scale(SU) -> Quantized GEMV -> Scale(SV)
针对 batch_size=1 的加速：

第一步 (验证): 确认 decode_matvec_e8p 的耗时占比。使用 nsight-systems 或 torch.cuda.Event 测量。如果占比 < 30%，则瓶颈在 Python overhead，优先上 CUDA Graph。如果占比 > 50%，则必须重写 Kernel。

耗时占比测试如下：
=== 分段耗时分析 ===
1. 前处理 (Hadamard + Scale): 76.9461 ms (49.8%)
2. 核心 (Decode E8P):         1.1444 ms (0.7%)
3. 后处理 (Hadamard + Scale): 76.2826 ms (49.4%)
总耗时:                       154.3731 ms

主要的性能瓶颈在于matmul_had_cuda这个函数，对于这个函数有很多强制转换以及CPU2GPU的PCIe数据传输



第二步 (简化): 你的 QuantizedE8P12Linear 类中，SU 和 SV 的乘法是独立的。立即将它们手动融合到 Python 代码中（如果可能），或者准备在 Triton Kernel 中融合。
第三步 (实现): 编写 Triton GEMV Kernel。
重点: 将 codebook 放入 Shared Memory。
重点: 确保输入向量 x 只被读取一次（放入 Shared Memory）。
难点: 移植 get_full_grid 的位运算逻辑到 Triton。建议先在 Python 中写一个纯 Tensor 操作的版本（不含循环），验证正确性后，直接翻译成 Triton DSL。

针对 batch_size>1的加速:


解决lora加载的问题


3/4 针对Deepseek模型进行Hessian矩阵生成
现在的问题是维度做不了hadamard矩阵变换，思路是拆分成几个小矩阵然后每个小矩阵生成hadamard变换，blockwise hadamard
deepseek这块主要是有两个维度，1408这个维度hadamard矩阵没法处理


3/7 开始系统的设计
1. 首先是基础设施的搭建与baseline的实现
目标让2bitmixtral跑起来，实现最基础的同步expert offloading (不带复杂的Cache)
技术：
- 内存与显存池的初始化 (memory manegement) CPU端侧必须使用pinned memory来存储2bit专家权重, GPU端侧预先分配固定大小VRAM pool,用于存放当前计算所需要的专家
- 2 bit反量化算法开发, W2A16
- 朴素的执行引擎 Naive Execution Engine
