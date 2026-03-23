"""
简化版比较脚本 - 顺序加载，避免OOM
1. 加载并测试HF模型
2. 删除HF模型
3. 加载并测试自定义模型
4. 比较输出
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.deepseek_moe import DeepseekForCausalLM, DeepseekConfig
import argparse
import gc
import json

def print_layer_stats(name, tensor):
    """打印张量的统计信息"""
    stats = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'mean': float(tensor.mean()),
        'std': float(tensor.std()),
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'has_nan': bool(torch.isnan(tensor).any()),
        'has_inf': bool(torch.isinf(tensor).any()),
    }
    return stats


def create_causal_mask(seq_len, device, dtype):
    """创建因果mask"""
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device)
    attention_mask.masked_fill_(causal_mask, torch.finfo(dtype).min)
    return attention_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-ai/deepseek-moe-16b-base', type=str)
    parser.add_argument('--num_layers', default=10, type=int)
    parser.add_argument('--text', default='The capital of France is', type=str)
    args = parser.parse_args()
    
    print("="*100)
    print("DEEPSEEK MODEL COMPARISON - Simple Sequential Loading")
    print("="*100)
    
    # 加载tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 准备输入
    print(f"[2/5] Preparing input: '{args.text}'")
    inputs = tokenizer(args.text, return_tensors='pt')
    input_ids = inputs['input_ids']
    seq_len = input_ids.shape[1]
    print(f"      Input shape: {input_ids.shape}, seq_len: {seq_len}")
    
    # ===== STEP 1: HF 模型 =====
    print(f"\n[3/5] Loading and testing HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    hf_model.eval()
    
    # 获取embedding层的设备（通常是cuda:0）
    embed_device = next(hf_model.model.embed_tokens.parameters()).device
    input_ids = input_ids.to(embed_device)
    
    # 准备mask和position_ids（基于embedding设备）
    attention_mask = create_causal_mask(seq_len, embed_device, torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=embed_device).unsqueeze(0)
    
    print(f"      Embedding device: {embed_device}")
    print(f"      Attention mask shape: {attention_mask.shape}")
    
    hf_outputs = {}
    
    with torch.no_grad():
        # Embedding
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        hf_outputs['embedding'] = print_layer_stats('embedding', hf_hidden)
        print(f"      Embedding output: {hf_outputs['embedding']}")
        
        # 逐层处理
        for layer_idx in range(min(args.num_layers, len(hf_model.model.layers))):
            print(f"      HF Layer {layer_idx}...", end='', flush=True)
            layer = hf_model.model.layers[layer_idx]
            layer_device = next(layer.parameters()).device
            
            # 确保hidden_states在当前层的设备上
            hf_hidden = hf_hidden.to(layer_device)
            attn_mask_layer = attention_mask.to(layer_device)
            pos_ids_layer = position_ids.to(layer_device)
            
            residual = hf_hidden
            hf_hidden = layer.input_layernorm(hf_hidden)
            attn_out = layer.self_attn(hf_hidden, attention_mask=attn_mask_layer, position_ids=pos_ids_layer)[0]
            hf_hidden = residual + attn_out
            
            residual = hf_hidden
            hf_hidden = layer.post_attention_layernorm(hf_hidden)
            mlp_out = layer.mlp(hf_hidden)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            hf_hidden = residual + mlp_out
            
            hf_outputs[f'layer_{layer_idx}'] = print_layer_stats(f'hf_layer_{layer_idx}', hf_hidden)
            print(f" mean={hf_outputs[f'layer_{layer_idx}']['mean']:.6f}")
        
        hf_hidden_final = hf_model.model.norm(hf_hidden)
        hf_outputs['norm'] = print_layer_stats('hf_norm', hf_hidden_final)
    
    # 获取norm层的设备用于自定义模型
    norm_device = next(hf_model.model.norm.parameters()).device
    print(f"\n      HF model processing complete. Final device: {norm_device}")
    del hf_model, hf_hidden, hf_hidden_final
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    print("      HF model deleted and memory cleared.")
    
    # ===== STEP 2: 自定义模型 =====
    print(f"\n[4/5] Loading and testing custom model...")
    
    # 用device_map直接加载HF模型，然后复制权重给自定义模型
    print("      Loading HF model to extract weights...")
    hf_temp = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    custom_config = DeepseekConfig(**config.to_dict())
    custom_model = DeepseekForCausalLM(custom_config)
    
    print("      Copying weights...")
    custom_model.load_state_dict(hf_temp.state_dict(), strict=False)
    del hf_temp
    gc.collect()
    
    # 移到norm_device（和HF模型的norm层同一设备）
    print(f"      Moving custom model to {norm_device}...")
    custom_model = custom_model.bfloat16()
    custom_model = custom_model.to(norm_device)
    custom_model.eval()
    
    # 为自定义模型准备输入
    input_ids_custom = input_ids.to(next(custom_model.model.embed_tokens.parameters()).device)
    
    custom_outputs = {}
    
    with torch.no_grad():
        # Embedding
        custom_hidden = custom_model.model.embed_tokens(input_ids_custom)
        custom_outputs['embedding'] = print_layer_stats('embedding', custom_hidden)
        print(f"      Embedding output: {custom_outputs['embedding']}")
        
        # 逐层处理
        for layer_idx in range(min(args.num_layers, len(custom_model.model.layers))):
            print(f"      Custom Layer {layer_idx}...", end='', flush=True)
            layer = custom_model.model.layers[layer_idx]
            
            residual = custom_hidden
            custom_hidden = layer.input_layernorm(custom_hidden)
            attn_out = layer.self_attn(custom_hidden, attention_mask=attention_mask.to(custom_hidden.device), position_ids=position_ids.to(custom_hidden.device))[0]
            custom_hidden = residual + attn_out
            
            residual = custom_hidden
            custom_hidden = layer.post_attention_layernorm(custom_hidden)
            mlp_out = layer.mlp(custom_hidden)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            custom_hidden = residual + mlp_out
            
            custom_outputs[f'layer_{layer_idx}'] = print_layer_stats(f'custom_layer_{layer_idx}', custom_hidden)
            print(f" mean={custom_outputs[f'layer_{layer_idx}']['mean']:.6f}")
        
        custom_hidden_final = custom_model.model.norm(custom_hidden)
        custom_outputs['norm'] = print_layer_stats('custom_norm', custom_hidden_final)
    
    print("\n      Custom model processing complete.")
    
    # ===== STEP 3: 比较输出 =====
    print(f"\n[5/5] Comparing outputs...")
    print("\n" + "="*100)
    print(f"{'Layer':<20} {'HF Mean':<15} {'Custom Mean':<15} {'Diff':<15} {'Status':<20}")
    print("="*100)
    
    for layer_name in hf_outputs.keys():
        if layer_name in custom_outputs:
            hf_mean = hf_outputs[layer_name]['mean']
            custom_mean = custom_outputs[layer_name]['mean']
            diff = abs(hf_mean - custom_mean)
            
            if diff < 0.01:
                status = "✓ Match"
            elif diff < 0.1:
                status = "⚠ Close"
            else:
                status = "✗ Diverge"
            
            print(f"{layer_name:<20} {hf_mean:<15.6f} {custom_mean:<15.6f} {diff:<15.6f} {status:<20}")
    
    print("="*100)


if __name__ == "__main__":
    main()
