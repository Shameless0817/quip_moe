"""
简化版对比脚本 - 比较 HuggingFace 和自定义模型的前5层输出
支持多GPU自动分布
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.deepseek_moe import DeepseekForCausalLM, DeepseekConfig
import argparse
import gc

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-ai/deepseek-moe-16b-base', type=str)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--text', default='The capital of France is', type=str)
    args = parser.parse_args()
    
    print("="*100)
    print("DEEPSEEK MODEL COMPARISON - Simplified Version")
    print("="*100)
    
    # 加载tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 准备输入
    print(f"[2/3] Preparing input: '{args.text}'")
    inputs = tokenizer(args.text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
    print(f"      Input shape: {input_ids.shape}")
    
    # 加载 HF 模型
    print(f"\n[3/3] Loading models...")
    print("      Loading HF model with auto device_map (uses all GPUs)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分布到所有可用GPU
        trust_remote_code=True
    )
    hf_model.eval()
    
    # 获取主设备
    main_device = next(hf_model.parameters()).device
    input_ids = input_ids.to(main_device)
    attention_mask = attention_mask.to(main_device)
    
    print(f"      HF model loaded on device: {main_device}")
    print(f"      Loading custom model on same device...")
    
    # 加载自定义模型并复制权重
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    custom_config = DeepseekConfig(**config.to_dict())
    custom_model = DeepseekForCausalLM(custom_config).to(main_device).bfloat16()
    custom_model.eval()
    
    # 简单的权重复制
    print("      Copying weights...")
    hf_state_dict = hf_model.state_dict()
    custom_model.load_state_dict(hf_state_dict, strict=False)
    
    # 提取并比较每一层的输出
    print("\n" + "="*100)
    print(f"{'Layer':<30} {'HF Mean':<12} {'Custom Mean':<12} {'Diff':<12} {'Status':<15}")
    print("="*100)
    
    hf_outputs = []
    custom_outputs = []
    
    torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        # ===== HuggingFace 模型 =====
        print("\nExtracting HF model outputs...")
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        hf_outputs.append(print_layer_stats('embedding', hf_hidden))
        
        for layer_idx in range(min(args.num_layers, len(hf_model.model.layers))):
            print(f"  Layer {layer_idx}...", end='\r')
            layer = hf_model.model.layers[layer_idx]
            
            # 原始forward逻辑
            residual = hf_hidden
            hf_hidden = layer.input_layernorm(hf_hidden)
            attn_out = layer.self_attn(hf_hidden, attention_mask=attention_mask)[0]
            hf_hidden = residual + attn_out
            
            residual = hf_hidden
            hf_hidden = layer.post_attention_layernorm(hf_hidden)
            mlp_out = layer.mlp(hf_hidden)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            hf_hidden = residual + mlp_out
            
            hf_outputs.append(print_layer_stats(f'layer_{layer_idx}', hf_hidden))
        
        # 保存HF输出到CPU，清除GPU缓存
        hf_outputs_cpu = []
        for output in hf_outputs:
            hf_outputs_cpu.append(output)
        hf_outputs = hf_outputs_cpu
        
        del hf_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # ===== 自定义模型 =====
        print("\nExtracting custom model outputs...")
        custom_hidden = custom_model.model.embed_tokens(input_ids)
        custom_outputs.append(print_layer_stats('embedding', custom_hidden))
        
        for layer_idx in range(min(args.num_layers, len(custom_model.model.layers))):
            print(f"  Layer {layer_idx}...", end='\r')
            layer = custom_model.model.layers[layer_idx]
            
            # 确保在正确设备
            layer_device = next(layer.parameters()).device
            custom_hidden = custom_hidden.to(layer_device)
            attention_mask_device = attention_mask.to(layer_device)
            
            # forward逻辑
            residual = custom_hidden
            custom_hidden = layer.input_layernorm(custom_hidden)
            attn_out = layer.self_attn(custom_hidden, attention_mask=attention_mask_device)[0]
            custom_hidden = residual + attn_out
            
            residual = custom_hidden
            custom_hidden = layer.post_attention_layernorm(custom_hidden)
            mlp_out = layer.mlp(custom_hidden)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            custom_hidden = residual + mlp_out
            
            custom_outputs.append(print_layer_stats(f'layer_{layer_idx}', custom_hidden))
        
        # 清理
        del custom_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # ===== 打印对比结果 =====
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)
    print(f"{'Layer':<30} {'HF Mean':<15} {'Custom Mean':<15} {'Diff':<15} {'Status':<15}")
    print("-"*100)
    
    max_divergence = 0
    first_diverge_layer = None
    
    for i, (hf_out, custom_out) in enumerate(zip(hf_outputs, custom_outputs)):
        layer_name = hf_out['name']
        hf_mean = hf_out['mean']
        custom_mean = custom_out['mean']
        diff = abs(hf_mean - custom_mean)
        max_divergence = max(max_divergence, diff)
        
        # 检测异常
        status = "✓ OK"
        if hf_out['has_nan'] or custom_out['has_nan']:
            status = "⚠️  NaN!"
        elif hf_out['has_inf'] or custom_out['has_inf']:
            status = "⚠️  Inf!"
        elif diff > 1.0:
            status = "❌ HIGH DIFF"
            if first_diverge_layer is None:
                first_diverge_layer = layer_name
        elif diff > 0.1:
            status = "⚠️  MED DIFF"
            if first_diverge_layer is None:
                first_diverge_layer = layer_name
        
        print(f"{layer_name:<30} {hf_mean:<15.6f} {custom_mean:<15.6f} {diff:<15.6f} {status:<15}")
    
    # 总结
    print("="*100)
    print("\nSUMMARY")
    print("-"*100)
    print(f"Max divergence: {max_divergence:.6f}")
    if first_diverge_layer:
        print(f"First divergence at: {first_diverge_layer}")
    
    if max_divergence < 0.01:
        print("✓ Models are nearly identical!")
    elif max_divergence < 0.1:
        print("⚠️  Minor divergence (may be dtype/RNG related)")
    elif max_divergence < 1.0:
        print("⚠️  Moderate divergence - check weight loading")
    else:
        print("❌ High divergence - serious weight loading issue")
    
    print("\nDiagnostic checklist:")
    print("  [ ] Check weight dtype (bf16 vs fp32)")
    print("  [ ] Verify layer norm parameters")
    print("  [ ] Check attention weight loading")
    print("  [ ] Check MLP/MoE weight loading")
    print("  [ ] Look for uninitialized parameters")
    print()

if __name__ == '__main__':
    main()
