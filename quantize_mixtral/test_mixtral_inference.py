#!/usr/bin/env python3
"""
Mixtral 量化模型推理测试脚本

用法:
    python test_mixtral_inference.py --model_path /path/to/hf_model --prompt "你的提示词"
"""

import argparse
import os
import sys
import time
import json
import gc

# 将父目录添加到 Python 路径（如果需要导入自定义模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoConfig

# 导入自定义的 Mixtral 模型和量化相关模块
from model.mixtral_moe import MixtralForCausalLM
from lib import codebook
from lib.linear import QuantizedLinear, FusedQuantizedLinear

torch.set_grad_enabled(False)


def clear_memory():
    """清理所有 GPU 缓存"""
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


def print_gpu_memory(prefix=""):
    """打印 GPU 显存使用情况"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"{prefix}GPU {i}: {allocated:.2f}/{total:.1f} GB (预留: {reserved:.2f} GB)")


def parse_args():
    parser = argparse.ArgumentParser(description='Mixtral 量化模型推理测试')
    parser.add_argument('--model_path', type=str, required=True,
                        help='HuggingFace 格式模型的路径')
    parser.add_argument('--prompt', type=str, default='It is a truth universally acknowledged that',
                        help='推理使用的提示词')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='生成的最大 token 数量')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p 采样参数')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k 采样参数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备 (cuda/cpu)')
    parser.add_argument('--do_sample', action='store_true',
                        help='是否使用采样（否则使用贪婪解码）')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能基准测试')
    return parser.parse_args()


def load_quantized_model(model_path, device='cuda'):
    """
    加载量化后的 Mixtral 模型
    """
    print(f"正在从 {model_path} 加载模型...")
    start_time = time.time()
    
    # 清理显存
    clear_memory()
    print("初始显存状态:")
    if device == 'cuda':
        print_gpu_memory("  ")
    
    # 加载配置
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 创建模型配置
    config = AutoConfig.from_pretrained(model_path)
    
    # 确保 quip_params 被正确加载
    if 'quip_params' in config_dict:
        config.quip_params = config_dict['quip_params']
    
    # 初始化模型结构（在CPU上）
    print("初始化模型结构（CPU）...")
    model = MixtralForCausalLM(config)
    
    # 加载权重
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(weights_path):
        print("加载模型权重...")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        # 立即删除state_dict释放内存
        del state_dict
        clear_memory()
    else:
        # 检查是否有分片权重
        shard_files = [f for f in os.listdir(model_path) if f.startswith('pytorch_model-') and f.endswith('.bin')]
        if shard_files:
            print(f"加载分片权重 ({len(shard_files)} 个文件)...")
            for shard_file in sorted(shard_files):
                shard_path = os.path.join(model_path, shard_file)
                shard_dict = torch.load(shard_path, map_location='cpu', weights_only=False)
                model.load_state_dict(shard_dict, strict=False)
                del shard_dict
                clear_memory()
                print(f"  已加载 {shard_file}")
        else:
            raise FileNotFoundError(f"在 {model_path} 中找不到模型权重文件")
    
    # 先转换为半精度（在CPU上）
    print("转换为 FP16...")
    model = model.half()
    
    # 再移动到设备
    if device == 'cuda':
        print(f"移动模型到 {device}...")
        print("移动前显存:")
        print_gpu_memory("  ")
        
    model = model.to(device)
    model.eval()
    
    if device == 'cuda':
        print("移动后显存:")
        print_gpu_memory("  ")
    
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时 {load_time:.2f} 秒")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e9:.2f}B")
    
    return model


def load_tokenizer(model_path):
    """
    加载 tokenizer
    """
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_new_tokens=128, 
                  temperature=0.7, top_p=0.9, top_k=50, 
                  do_sample=True, device='cuda'):
    """
    生成文本 - 自定义生成逻辑，支持多GPU模型
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    
    # 检测模型是否在多个GPU上
    is_multi_gpu = check_is_multi_gpu(model)
    
    if is_multi_gpu:
        # 多GPU：输入放在第一层所在的设备
        first_device = next(model.model.embed_tokens.parameters()).device
        input_ids = inputs['input_ids'].to(first_device)
        attention_mask = inputs['attention_mask'].to(first_device)
    else:
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
    
    print(f"\n输入 prompt: {prompt}")
    print(f"输入 token 数量: {input_ids.shape[1]}")
    print(f"使用模式: {'多GPU' if is_multi_gpu else '单GPU'}")
    print("-" * 50)
    
    # 记录生成时间
    start_time = time.time()
    
    # 逐token生成
    generated_ids = input_ids.clone()
    
    for step in range(max_new_tokens):
        try:
            # 前向传播
            if is_multi_gpu:
                logits = forward_multi_gpu(model, generated_ids, attention_mask)
            else:
                # 单GPU模型：直接调用
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                logits = outputs.logits
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n警告: GPU 显存不足，清理缓存...")
                clear_memory()
                continue
            elif "device" in str(e).lower() and "cuda" in str(e).lower():
                print(f"\n警告: 设备错误 {e}，尝试修复...")
                # 重新检查并修复设备
                if is_multi_gpu:
                    hidden_states = model.model.embed_tokens(generated_ids.to(next(model.model.embed_tokens.parameters()).device))
                continue
            else:
                raise
        
        # 只取最后一个位置的logits
        next_token_logits = logits[:, -1, :]
        
        # 采样或贪婪解码
        if do_sample:
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # Top-k 过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将排序后的索引映射回原始索引
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # 贪婪解码
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 拼接新token - 确保在正确的设备上
        if is_multi_gpu:
            next_token = next_token.to(input_ids.device)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # 更新attention_mask
        if attention_mask is not None:
            new_mask = torch.ones((attention_mask.shape[0], 1), 
                                 device=attention_mask.device, 
                                 dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
        
        # 检查是否生成了EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    generation_time = time.time() - start_time
    
    # 解码输出
    generated_tokens = generated_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 计算统计信息
    num_generated_tokens = len(generated_tokens)
    tokens_per_second = num_generated_tokens / generation_time if generation_time > 0 else 0
    
    return {
        'generated_text': generated_text,
        'full_output': full_output,
        'num_generated_tokens': num_generated_tokens,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second,
    }


def check_is_multi_gpu(model):
    """
    检查模型是否分布在多个GPU上
    """
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        return False
    
    if len(model.model.layers) == 0:
        return False
    
    devices = set()
    devices.add(next(model.model.embed_tokens.parameters()).device.type)
    
    for layer in model.model.layers[:min(3, len(model.model.layers))]:  # 只检查前3层
        devices.add(next(layer.parameters()).device.type)
    
    # 如果有device index不同的cuda设备，则为多GPU
    if 'cuda' in devices:
        cuda_indices = set()
        try:
            cuda_indices.add(next(model.model.embed_tokens.parameters()).device.index)
            for layer in model.model.layers[:min(3, len(model.model.layers))]:
                cuda_indices.add(next(layer.parameters()).device.index)
            return len(cuda_indices) > 1
        except:
            return False
    
    return len(devices) > 1


def forward_multi_gpu(model, input_ids, attention_mask=None):
    """
    多GPU模型的前向传播 - 完整重写
    自动处理各层间的设备转移
    """
    # 获取embed所在的设备
    embed_device = next(model.model.embed_tokens.parameters()).device
    input_ids = input_ids.to(embed_device)
    
    # Embedding层
    hidden_states = model.model.embed_tokens(input_ids)
    
    # 获取所有layer及其设备映射
    layers_devices = {}
    for idx, layer in enumerate(model.model.layers):
        layer_device = next(layer.parameters()).device
        layers_devices[idx] = layer_device
    
    # 遍历所有decoder层
    for layer_idx, layer in enumerate(model.model.layers):
        layer_device = layers_devices[layer_idx]
        
        # 将hidden_states移动到该层的设备
        hidden_states = hidden_states.to(layer_device)
        
        # 将attention_mask也移动到该设备
        if attention_mask is not None:
            layer_attention_mask = attention_mask.to(layer_device)
        else:
            layer_attention_mask = None
        
        try:
            # 调用decoder layer - 使用最小必要参数
            layer_outputs = layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                use_cache=False,
            )
            
            # 提取hidden_states（DecoderLayer返回元组）
            if isinstance(layer_outputs, tuple) or isinstance(layer_outputs, list):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
                
        except TypeError as e:
            # 如果参数不对，尝试只传hidden_states
            print(f"第{layer_idx}层参数错误，尝试简化参数: {e}")
            layer_outputs = layer(
                hidden_states,
                attention_mask=layer_attention_mask,
            )
            if isinstance(layer_outputs, tuple) or isinstance(layer_outputs, list):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
    
    # Norm层
    norm_device = next(model.model.norm.parameters()).device
    hidden_states = hidden_states.to(norm_device)
    hidden_states = model.model.norm(hidden_states)
    
    # LM head
    lm_head_device = next(model.lm_head.parameters()).device
    hidden_states = hidden_states.to(lm_head_device)
    logits = model.lm_head(hidden_states)
    
    return logits


@torch.inference_mode()
def run_benchmark(model, tokenizer, device='cuda', num_runs=5):
    """
    运行性能基准测试
    """
    print("\n" + "=" * 50)
    print("性能基准测试")
    print("=" * 50)
    
    test_prompts = [
        "The quick brown fox",
        "In a galaxy far far away",
        "Once upon a time in a land of magic and wonder",
        "The fundamental principles of quantum mechanics state that",
    ]
    
    results = []
    
    for prompt in test_prompts:
        print(f"\n测试 prompt: '{prompt[:50]}...' (长度: {len(prompt)})")
        
        prompt_results = []
        for run in range(num_runs):
            result = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=64,
                do_sample=False,  # 使用贪婪解码以获得一致的结果
                device=device
            )
            prompt_results.append(result['tokens_per_second'])
            print(f"  Run {run + 1}: {result['tokens_per_second']:.2f} tokens/s")
        
        avg_speed = sum(prompt_results) / len(prompt_results)
        results.append(avg_speed)
        print(f"  平均速度: {avg_speed:.2f} tokens/s")
    
    overall_avg = sum(results) / len(results)
    print("\n" + "-" * 50)
    print(f"总体平均速度: {overall_avg:.2f} tokens/s")
    
    return overall_avg


@torch.inference_mode()
def interactive_mode(model, tokenizer, device='cuda'):
    """
    交互式对话模式
    """
    print("\n" + "=" * 50)
    print("交互式对话模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清除对话历史")
    print("=" * 50)
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = ""
                print("对话历史已清除")
                continue
            
            if not user_input:
                continue
            
            # 构建 prompt
            if conversation_history:
                prompt = f"{conversation_history}\n用户: {user_input}\n助手:"
            else:
                prompt = f"用户: {user_input}\n助手:"
            
            # 生成回复
            result = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                device=device
            )
            
            response = result['generated_text'].strip()
            
            # 截取到第一个 "用户:" 之前（如果有的话）
            if "用户:" in response:
                response = response.split("用户:")[0].strip()
            
            print(f"\n助手: {response}")
            print(f"(生成 {result['num_generated_tokens']} tokens, "
                  f"耗时 {result['generation_time']:.2f}s, "
                  f"{result['tokens_per_second']:.2f} tokens/s)")
            
            # 更新对话历史
            conversation_history = f"{prompt} {response}"
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break


def main():
    args = parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 张 GPU")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    print("\n选择模型加载策略...")
    
    if torch.cuda.device_count() > 1 and args.device == 'cuda':
        print("使用多GPU加载策略...")
        try:
            model = load_quip_model_multi_gpu(args.model_path)
            use_multi_gpu_forward = True
        except Exception as e:
            print(f"多GPU加载失败: {e}")
            print("尝试单GPU加载...")
            model = load_quantized_model(args.model_path, device=args.device)
            use_multi_gpu_forward = False
    else:
        print("使用单GPU加载策略...")
        model = load_quantized_model(args.model_path, device=args.device)
        use_multi_gpu_forward = False
    
    tokenizer = load_tokenizer(args.model_path)
    
    if args.benchmark:
        # 运行基准测试
        run_benchmark(model, tokenizer, device=args.device)
    else:
        # 单次推理测试
        result = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            device=args.device
        )
        
        print("\n生成结果:")
        print("=" * 50)
        print(result['generated_text'])
        print("=" * 50)
        print(f"\n统计信息:")
        print(f"  生成 token 数: {result['num_generated_tokens']}")
        print(f"  生成耗时: {result['generation_time']:.2f} 秒")
        print(f"  生成速度: {result['tokens_per_second']:.2f} tokens/秒")
        
        # 询问是否进入交互模式
        try:
            response = input("\n是否进入交互模式？(y/n): ").strip().lower()
            if response == 'y':
                interactive_mode(model, tokenizer, device=args.device)
        except KeyboardInterrupt:
            print("\n程序已退出")
        except Exception as e:
            print(f"\n发生错误: {e}")


def load_quip_model_multi_gpu(model_path):
    """
    加载 QuIP# 量化模型到多 GPU
    使用逐层加载策略，避免一次性占用过多显存
    """
    print(f"\n{'='*60}")
    print(f"加载模型: {model_path}")
    print(f"{'='*60}")
    
    clear_memory()
    print("\n[1] 初始显存状态:")
    print_gpu_memory()
    
    # 加载配置
    print("\n[2] 加载配置...")
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    config = AutoConfig.from_pretrained(model_path)
    if 'quip_params' in config_dict:
        config.quip_params = config_dict['quip_params']
        print(f"    量化配置: codebook={config.quip_params.get('codebook')}")
    
    num_gpus = torch.cuda.device_count()
    num_layers = config.num_hidden_layers
    
    # 计算每张卡分配多少层
    # 注意：第一张卡还要放 embed，最后一张卡要放 lm_head
    # 所以中间的卡可以多放几层
    layers_per_gpu = num_layers // num_gpus
    extra = num_layers % num_gpus
    
    layer_assignment = []
    current = 0
    for gpu_id in range(num_gpus):
        n = layers_per_gpu + (1 if gpu_id < extra else 0)
        layer_assignment.append((current, current + n, gpu_id))
        current += n
    
    print(f"\n[3] 层分配计划:")
    print(f"    总层数: {num_layers}, GPU 数: {num_gpus}")
    for start, end, gpu_id in layer_assignment:
        print(f"    GPU {gpu_id}: layers {start}-{end-1} ({end-start} 层)")
    
    # 加载 state_dict 到 CPU
    print("\n[4] 加载权重到 CPU...")
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    print(f"    加载完成，共 {len(state_dict)} 个张量")
    
    # 在 CPU 上初始化模型（使用 meta tensor 避免分配内存）
    print("\n[5] 初始化模型结构...")
    
    # 方法：直接在目标设备上创建模型的各个部分
    
    # 先创建一个空模型结构用于获取参数名
    model = MixtralForCausalLM(config)
    
    # 移动 embed_tokens 到 GPU 0
    print("\n[6] 分配模型到各 GPU...")
    print("    移动 embed_tokens 到 GPU 0...")
    model.model.embed_tokens = model.model.embed_tokens.half().to('cuda:0')
    
    # 加载 embed_tokens 权重
    embed_keys = [k for k in state_dict.keys() if 'embed_tokens' in k]
    for key in embed_keys:
        param_name = key.replace('model.', '', 1) if key.startswith('model.') else key
        # 找到对应的参数并加载
        parts = key.split('.')
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        tensor = state_dict[key].half().to('cuda:0')
        if hasattr(obj, parts[-1]):
            param = getattr(obj, parts[-1])
            if isinstance(param, torch.nn.Parameter):
                param.data = tensor
            else:
                setattr(obj, parts[-1], tensor)
    
    # 删除已加载的权重释放 CPU 内存
    for key in embed_keys:
        del state_dict[key]
    
    clear_memory()
    print_gpu_memory("    ")
    
    # 逐层移动 transformer 层
    print("\n    移动 transformer 层...")
    for layer_idx in range(num_layers):
        # 确定这一层应该放在哪张卡
        target_gpu = None
        for start, end, gpu_id in layer_assignment:
            if start <= layer_idx < end:
                target_gpu = gpu_id
                break
        
        device = f'cuda:{target_gpu}'
        
        # 先转换为半精度再移动（节省显存）
        model.model.layers[layer_idx] = model.model.layers[layer_idx].half()
        # 移动层到目标 GPU - 这会递归移动所有子模块
        model.model.layers[layer_idx] = model.model.layers[layer_idx].to(device)
        
        # 加载这一层的权重
        layer_keys = [k for k in list(state_dict.keys()) if f'layers.{layer_idx}.' in k]
        for key in layer_keys:
            parts = key.split('.')
            obj = model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            
            tensor = state_dict[key].half().to(device)
            if hasattr(obj, parts[-1]):
                param = getattr(obj, parts[-1])
                if isinstance(param, torch.nn.Parameter):
                    param.data = tensor
                else:
                    setattr(obj, parts[-1], tensor)
            
            # 立即删除 CPU 上的权重
            del state_dict[key]
        
        # 验证设备分配
        for name, param in model.model.layers[layer_idx].named_parameters():
            if param.device.type != 'cuda' or param.device.index != target_gpu:
                param.data = param.data.to(device)
        
        if (layer_idx + 1) % 8 == 0:
            clear_memory()
            print(f"    已加载 {layer_idx + 1}/{num_layers} 层")
            print_gpu_memory("      ")
    
    # 移动 norm 和 lm_head 到最后一张卡
    last_gpu = f'cuda:{num_gpus - 1}'
    print(f"\n    移动 norm 和 lm_head 到 GPU {num_gpus - 1}...")
    
    model.model.norm = model.model.norm.half().to(last_gpu)
    model.lm_head = model.lm_head.half().to(last_gpu)
    
    # 加载剩余权重
    for key in list(state_dict.keys()):
        parts = key.split('.')
        obj = model
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        
        if 'norm' in key or 'lm_head' in key:
            device = last_gpu
        else:
            device = 'cuda:0'
        
        tensor = state_dict[key].half().to(device)
        if hasattr(obj, parts[-1]):
            param = getattr(obj, parts[-1])
            if isinstance(param, torch.nn.Parameter):
                param.data = tensor
            else:
                setattr(obj, parts[-1], tensor)
        
        del state_dict[key]
    
    del state_dict
    clear_memory()
    
    model.eval()
    
    print(f"\n[7] 最终显存使用:")
    print_gpu_memory("    ")
    
    # 记录设备映射
    model._device_map = {
        'embed_tokens': 'cuda:0',
        'layers': layer_assignment,
        'norm': last_gpu,
        'lm_head': last_gpu,
    }
    
    return model


def load_model_simple(model_path):
    """
    更简单的加载方式 - 使用 PyTorch 的模型并行
    """
    print(f"\n{'='*60}")
    print(f"简单多 GPU 加载: {model_path}")
    print(f"{'='*60}")
    
    clear_memory()
    
    # 加载配置
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    config = AutoConfig.from_pretrained(model_path)
    if 'quip_params' in config_dict:
        config.quip_params = config_dict['quip_params']
    
    num_gpus = torch.cuda.device_count()
    num_layers = config.num_hidden_layers
    
    print(f"\nGPU 数量: {num_gpus}")
    print(f"模型层数: {num_layers}")
    
    # 在 CPU 上创建模型
    print("\n在 CPU 上创建模型...")
    model = MixtralForCausalLM(config)
    
    # 加载权重
    print("加载权重...")
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # 直接加载，不做 strict 检查
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  缺失的键: {len(missing)}")
    if unexpected:
        print(f"  意外的键: {len(unexpected)}")
    
    del state_dict
    gc.collect()
    
    # 转为半精度（减少 CPU 内存占用）
    print("转换为 FP16...")
    model = model.half()
    
    # 计算每张卡放多少层
    layers_per_gpu = num_layers // num_gpus
    
    print(f"\n分配到 {num_gpus} 张 GPU，每张约 {layers_per_gpu} 层...")
    
    # 移动 embed_tokens
    model.model.embed_tokens = model.model.embed_tokens.to('cuda:0')
    print("  embed_tokens -> GPU 0")
    
    # 移动每一层
    for i in range(num_layers):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        model.model.layers[i] = model.model.layers[i].to(f'cuda:{gpu_id}')
        
        if (i + 1) % 8 == 0:
            print(f"  layers 0-{i} 已分配")
            gc.collect()
            torch.cuda.empty_cache()
    
    # 移动最后的层
    model.model.norm = model.model.norm.to(f'cuda:{num_gpus-1}')
    model.lm_head = model.lm_head.to(f'cuda:{num_gpus-1}')
    print(f"  norm, lm_head -> GPU {num_gpus-1}")
    
    model.eval()
    
    print("\n最终显存使用:")
    print_gpu_memory("  ")
    
    return model


class MultiGPUModelWrapper(torch.nn.Module):
    """
    包装模型以支持多 GPU 前向传播
    """
    def __init__(self, model, num_gpus):
        super().__init__()
        self.model = model
        self.num_gpus = num_gpus
        self.num_layers = len(model.model.layers)
        self.layers_per_gpu = self.num_layers // num_gpus
    
    def get_layer_device(self, layer_idx):
        gpu_id = min(layer_idx // self.layers_per_gpu, self.num_gpus - 1)
        return f'cuda:{gpu_id}'
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # embed 在 GPU 0
        input_ids = input_ids.to('cuda:0')
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        # 通过每一层
        for i, layer in enumerate(self.model.model.layers):
            # 获取当前层所在的设备
            layer_device = next(layer.parameters()).device
            
            # 移动 hidden_states 到该设备
            hidden_states = hidden_states.to(layer_device)
            
            # 移动 attention_mask（如果存在）
            if attention_mask is not None:
                attention_mask_device = attention_mask.to(layer_device)
                layer_output = layer(hidden_states, attention_mask=attention_mask_device)
            else:
                layer_output = layer(hidden_states)
            
            # 提取 hidden_states（可能是元组）
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
        
        # 最后的处理 - 获取norm所在的设备
        norm_device = next(self.model.model.norm.parameters()).device
        hidden_states = hidden_states.to(norm_device)
        hidden_states = self.model.model.norm(hidden_states)
        
        # lm_head
        lm_head_device = next(self.model.lm_head.parameters()).device
        hidden_states = hidden_states.to(lm_head_device)
        logits = self.model.lm_head(hidden_states)
        
        return logits
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=100, 
                 temperature=1.0, do_sample=True, top_k=50, top_p=0.9,
                 pad_token_id=None, eos_token_id=None, **kwargs):
        """简单的生成函数"""
        
        input_ids = input_ids.to('cuda:0')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda:0')
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(generated, attention_mask)
            
            # 只取最后一个 token 的 logits
            next_token_logits = logits[:, -1, :]
            
            if do_sample:
                # 应用温度
                next_token_logits = next_token_logits / temperature
                
                # Top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 移动到正确设备并拼接
            next_token = next_token.to('cuda:0')
            generated = torch.cat([generated, next_token], dim=1)
            
            # 更新 attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device='cuda:0', dtype=attention_mask.dtype)
                ], dim=1)
            
            # 检查是否结束
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return generated


if __name__ == '__main__':
    # 清理所有 GPU 缓存
    clear_memory()
    
    # 运行主函数
    main()