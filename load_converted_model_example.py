#!/usr/bin/env python3
"""
加载转换后的Mixtral E8P+LoRA模型的示例脚本

使用方式：
    python load_converted_model_example.py \
        --model_dir ./converted_model \
        --prompt "Hello, how are you?" \
        --max_tokens 50
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def load_converted_model(model_dir, device='cuda'):
    """
    加载转换后的模型
    
    Args:
        model_dir: 保存转换模型的目录（包含pytorch_model.pt和lora_config.json）
        device: 设备（'cuda'或'cpu'）
        
    Returns:
        model: 加载的模型
    """
    model_dir = Path(model_dir)
    
    print(f"从 {model_dir} 加载模型...")
    config_path = model_dir / "lora_config.json"
    state_dict_path = model_dir / "pytorch_model.pt"
    
    # 验证文件存在
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not state_dict_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {state_dict_path}")
    
    # 加载配置信息
    print("加载配置...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"  - LoRA秩: {config.get('lora_rank')}")
    print(f"  - 转换层数: {config.get('num_layers_converted')}/{config.get('num_layers_total')}")
    
    # 获取原始模型路径（用于加载模型结构）
    original_model_path = config.get(
        'original_model_path', 
        'mistralai/Mixtral-8x7B-Instruct-v0.1'
    )
    
    # 加载原始模型结构（将被state_dict替换）
    print(f"加载基础模型结构: {original_model_path}")
    print("  （这可能需要几分钟时间下载模型...）")
    
    model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    # 加载转换后的权重（包括LoRA参数）暂时加载到cpu
    print(f"加载转换后的权重...")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    
    # strict=False 允许某些键不匹配（可能是optimizer状态等）
    model.load_state_dict(state_dict, strict=False)
    
    print("✓ 模型加载完成")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="加载并测试转换后的Mixtral E8P+LoRA模型"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="转换后模型的保存目录"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="输入提示文本"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="生成的最大token数"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="使用的设备（cuda或cpu）"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus采样参数"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model, config = load_converted_model(args.model_dir, device=args.device)
    model.eval()
    
    # 加载tokenizer
    print("\n加载tokenizer...")
    original_model_path = config.get(
        'original_model_path',
        'mistralai/Mixtral-8x7B-Instruct-v0.1'
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    except:
        # 如果加载失败，尝试使用默认的Mixtral tokenizer
        print("无法从原始模型路径加载tokenizer，尝试使用默认tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
    
    # 测试生成
    print("\n" + "=" * 60)
    print("文本生成测试")
    print("=" * 60)
    
    print(f"输入: {args.prompt}\n")
    
    # 编码输入
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    
    # 生成
    print("生成中...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = tokenizer.decode(
        generated_ids[0], 
        skip_special_tokens=True
    )
    
    print(f"输出:\n{generated_text}\n")
    
    # 模型信息
    print("=" * 60)
    print("模型信息")
    print("=" * 60)
    print(f"✓ LoRA秩: {config.get('lora_rank')}")
    print(f"✓ 转换层数: {config.get('num_layers_converted')}/{config.get('num_layers_total')}")
    print(f"✓ 模型类型: {config.get('model_type')}")
    print(f"✓ 使用设备: {args.device}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 总参数量: {total_params/1e9:.2f}B")
    print(f"✓ 可训练参数: {trainable_params/1e6:.2f}M")
    
    return model, tokenizer


# 简化版本：直接导入使用
def quick_load(model_dir, device='cuda'):
    """快速加载模型（不进行测试）"""
    model, _ = load_converted_model(model_dir, device=device)
    return model


if __name__ == "__main__":
    model, tokenizer = main()
