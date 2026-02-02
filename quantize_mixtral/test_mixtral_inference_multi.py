#!/usr/bin/env python3
"""
Mixtral 量化模型多 GPU 分布式推理脚本

支持将模型分布到多张 GPU 上进行推理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import json
import time
from transformers import AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from model.mixtral_moe import MixtralForCausalLM

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Mixtral 多 GPU 推理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='HuggingFace 格式模型的路径')
    parser.add_argument('--prompt', type=str, default='Hello, how are you today?',
                        help='推理使用的提示词')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='生成的最大 token 数量')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--do_sample', action='store_true',
                        help='是否使用采样')
    return parser.parse_args()


def get_device_map(model_path, num_gpus=None):
    """
    自动计算设备映射，将模型层均匀分布到多张 GPU
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    print(f"检测到 {num_gpus} 张 GPU")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    
    # 加载配置获取层数
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    num_layers = config_dict.get('num_hidden_layers', 32)
    
    # 手动创建设备映射
    device_map = {}
    
    # 嵌入层放在第一张卡
    device_map['model.embed_tokens'] = 0
    
    # 均匀分配 transformer 层到各张卡
    layers_per_gpu = num_layers // num_gpus
    extra_layers = num_layers % num_gpus
    
    current_layer = 0
    for gpu_id in range(num_gpus):
        # 计算这张卡分配多少层
        n_layers = layers_per_gpu + (1 if gpu_id < extra_layers else 0)
        for _ in range(n_layers):
            device_map[f'model.layers.{current_layer}'] = gpu_id
            current_layer += 1
    
    # 最后的 norm 和 lm_head 放在最后一张卡
    device_map['model.norm'] = num_gpus - 1
    device_map['lm_head'] = num_gpus - 1
    
    print(f"\n设备映射:")
    print(f"  embed_tokens -> GPU 0")
    print(f"  layers 0-{num_layers-1} -> 均匀分布到 GPU 0-{num_gpus-1}")
    print(f"  norm, lm_head -> GPU {num_gpus - 1}")
    
    return device_map


def load_model_multi_gpu(model_path):
    """
    使用 accelerate 将模型加载到多张 GPU
    """
    print(f"\n正在从 {model_path} 加载模型...")
    start_time = time.time()
    
    # 加载配置
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    config = AutoConfig.from_pretrained(model_path)
    if 'quip_params' in config_dict:
        config.quip_params = config_dict['quip_params']
    
    # 获取设备映射
    device_map = get_device_map(model_path)
    
    # 方法1：使用 accelerate 的 load_checkpoint_and_dispatch
    print("\n初始化空模型...")
    with init_empty_weights():
        model = MixtralForCausalLM(config)
    
    # 查找权重文件
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    if not os.path.exists(weights_path):
        # 检查是否有 safetensors
        safetensors_path = os.path.join(model_path, 'model.safetensors')
        if os.path.exists(safetensors_path):
            weights_path = safetensors_path
        else:
            # 检查分片文件
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            if os.path.exists(index_path):
                weights_path = model_path  # 使用目录
            else:
                raise FileNotFoundError(f"找不到权重文件: {model_path}")
    
    print(f"从 {weights_path} 加载权重并分发到多 GPU...")
    
    model = load_checkpoint_and_dispatch(
        model,
        weights_path,
        device_map=device_map,
        dtype=torch.float16,
        no_split_module_classes=["MixtralDecoderLayer", "MixtralSparseMoeBlock"],
    )
    
    model.eval()
    
    load_time = time.time() - start_time
    print(f"\n模型加载完成，耗时 {load_time:.2f} 秒")
    
    # 打印每张卡的显存使用情况
    print("\nGPU 显存使用情况:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
    
    return model


def load_model_manual_split(model_path):
    """
    手动将模型分割到多张 GPU（备用方法）
    """
    print(f"\n正在从 {model_path} 加载模型（手动分割）...")
    start_time = time.time()
    
    # 加载配置
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    config = AutoConfig.from_pretrained(model_path)
    if 'quip_params' in config_dict:
        config.quip_params = config_dict['quip_params']
    
    num_gpus = torch.cuda.device_count()
    num_layers = config.num_hidden_layers
    layers_per_gpu = num_layers // num_gpus
    
    # 在 CPU 上初始化模型
    print("在 CPU 上初始化模型...")
    model = MixtralForCausalLM(config)
    
    # 加载权重到 CPU
    print("加载权重到 CPU...")
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    torch.cuda.empty_cache()
    
    # 转换为半精度
    model = model.half()
    
    # 手动移动各部分到不同 GPU
    print("分发模型到多张 GPU...")
    
    # 嵌入层
    model.model.embed_tokens = model.model.embed_tokens.to('cuda:0')
    
    # 分配 transformer 层
    for i, layer in enumerate(model.model.layers):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        model.model.layers[i] = layer.to(f'cuda:{gpu_id}')
        if (i + 1) % 8 == 0:
            print(f"  已移动 {i + 1}/{num_layers} 层")
    
    # 最后的层
    model.model.norm = model.model.norm.to(f'cuda:{num_gpus - 1}')
    model.lm_head = model.lm_head.to(f'cuda:{num_gpus - 1}')
    
    model.eval()
    
    load_time = time.time() - start_time
    print(f"\n模型加载完成，耗时 {load_time:.2f} 秒")
    
    # 打印显存使用
    print("\nGPU 显存使用情况:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {allocated:.2f} GB")
    
    return model


def load_model_with_device_map_auto(model_path):
    """
    使用 device_map="auto" 自动分配（最简单的方法）
    """
    print(f"\n正在从 {model_path} 加载模型（自动设备映射）...")
    start_time = time.time()
    
    # 加载配置
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    config = AutoConfig.from_pretrained(model_path)
    if 'quip_params' in config_dict:
        config.quip_params = config_dict['quip_params']
    
    # 计算每张卡可用的最大内存
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        # 保留 2GB 作为缓冲
        total = torch.cuda.get_device_properties(i).total_memory
        max_memory[i] = int(total * 0.85)  # 使用 85% 的显存
    max_memory['cpu'] = '64GB'  # CPU 内存作为后备
    
    print(f"最大显存配置: {max_memory}")
    
    # 使用空权重初始化
    with init_empty_weights():
        model = MixtralForCausalLM(config)
    
    # 推断设备映射
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["MixtralDecoderLayer"],
        dtype=torch.float16,
    )
    
    print(f"\n自动推断的设备映射:")
    # 统计每张卡分配了多少层
    gpu_layers = {}
    for name, device in device_map.items():
        if device not in gpu_layers:
            gpu_layers[device] = 0
        gpu_layers[device] += 1
    for device, count in sorted(gpu_layers.items()):
        print(f"  {device}: {count} 个模块")
    
    # 加载权重
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    
    model = load_checkpoint_and_dispatch(
        model,
        weights_path,
        device_map=device_map,
        dtype=torch.float16,
        no_split_module_classes=["MixtralDecoderLayer"],
    )
    
    model.eval()
    
    load_time = time.time() - start_time
    print(f"\n模型加载完成，耗时 {load_time:.2f} 秒")
    
    return model


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_new_tokens=128,
                  temperature=0.7, do_sample=True):
    """
    生成文本（支持多 GPU）
    """
    # 获取模型的第一个设备（嵌入层所在的设备）
    if hasattr(model, 'hf_device_map'):
        # 使用 accelerate 加载的模型
        first_device = list(model.hf_device_map.values())[0]
        if isinstance(first_device, int):
            first_device = f'cuda:{first_device}'
    else:
        # 手动分割的模型
        first_device = next(model.parameters()).device
    
    print(f"输入设备: {first_device}")
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(first_device)
    attention_mask = inputs['attention_mask'].to(first_device)
    
    print(f"输入 prompt: {prompt}")
    print(f"输入 token 数量: {input_ids.shape[1]}")
    print("-" * 50)
    
    start_time = time.time()
    
    # 生成配置
    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': do_sample,
    }
    
    if do_sample:
        gen_kwargs['temperature'] = temperature
        gen_kwargs['top_p'] = 0.9
        gen_kwargs['top_k'] = 50
    
    # 生成
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs
    )
    
    generation_time = time.time() - start_time
    
    # 解码
    generated_tokens = outputs[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    num_tokens = len(generated_tokens)
    tokens_per_sec = num_tokens / generation_time
    
    return {
        'text': generated_text,
        'num_tokens': num_tokens,
        'time': generation_time,
        'speed': tokens_per_sec,
    }


def interactive_mode(model, tokenizer):
    """
    交互式对话
    """
    print("\n" + "=" * 50)
    print("交互式模式 (输入 'quit' 退出)")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n你: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue
            
            result = generate_text(model, tokenizer, prompt, 
                                   max_new_tokens=256, do_sample=True)
            
            print(f"\n助手: {result['text']}")
            print(f"({result['num_tokens']} tokens, {result['time']:.2f}s, "
                  f"{result['speed']:.1f} tokens/s)")
            
        except KeyboardInterrupt:
            break
    
    print("\n再见!")


def main():
    args = parse_args()
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("错误: 没有可用的 GPU")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"可用 GPU 数量: {num_gpus}")
    
    # 尝试不同的加载方法
    try:
        # 方法 1: 使用 accelerate（推荐）
        model = load_model_multi_gpu(args.model_path)
    except Exception as e:
        print(f"\naccelerater 加载失败: {e}")
        print("尝试使用自动设备映射...")
        try:
            # 方法 2: 自动设备映射
            model = load_model_with_device_map_auto(args.model_path)
        except Exception as e2:
            print(f"\n自动设备映射失败: {e2}")
            print("尝试手动分割...")
            # 方法 3: 手动分割
            model = load_model_manual_split(args.model_path)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试推理
    print("\n" + "=" * 50)
    print("开始推理测试")
    print("=" * 50)
    
    result = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    
    print(f"\n生成结果:")
    print("=" * 50)
    print(result['text'])
    print("=" * 50)
    print(f"生成 {result['num_tokens']} tokens")
    print(f"耗时 {result['time']:.2f} 秒")
    print(f"速度 {result['speed']:.1f} tokens/秒")
    
    # 显存使用情况
    print("\n最终 GPU 显存使用:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {allocated:.1f}/{total:.1f} GB "
              f"(预留: {reserved:.1f} GB)")
    
    # 进入交互模式
    try:
        choice = input("\n进入交互模式? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_mode(model, tokenizer)
    except KeyboardInterrupt:
        pass
    
    print("\n完成!")


if __name__ == '__main__':
    main()