"""
直接对比 HuggingFace 和量化后的自定义模型
加载 quantize_finetune_deepseek_s.py 生成的混合模型权重
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.deepseek_moe import DeepseekForCausalLM
import argparse
import os
import copy

def copy_weights_to_custom_model(hf_model, custom_model, quantized_path, device):
    """将量化后的权重复制到自定义模型"""
    print("Copying weights from quantized model to custom model...")
    
    hf_dict = hf_model.state_dict()
    custom_state = custom_model.state_dict()
    
    # 首先尝试从 quantized_path 加载已量化的权重
    missing_keys = []
    for key in custom_state.keys():
        if key in hf_dict:
            try:
                custom_state[key].copy_(hf_dict[key])
            except Exception as e:
                print(f"Warning: Could not copy {key}: {e}")
                missing_keys.append(key)
    
    if missing_keys:
        print(f"Missing {len(missing_keys)} keys: {missing_keys[:5]}...")
    
    custom_model.load_state_dict(custom_state, strict=False)
    return custom_model

def extract_layer_outputs(
    model, 
    input_ids, 
    attention_mask=None, 
    num_layers=5, 
    model_type="hf",
    return_all_states=False
):
    """
    逐层提取模型输出，便于调试
    
    Args:
        model: 模型实例
        input_ids: 输入token ids
        attention_mask: 注意掩码
        num_layers: 提取前几层
        model_type: "hf" 或 "custom"
        return_all_states: 是否返回所有中间状态
    
    Returns:
        outputs: 每层的输出统计信息
    """
    outputs = []
    all_hidden_states = []
    
    with torch.no_grad():
        # Embedding 层
        embed_device = model.model.embed_tokens.weight.device
        input_ids_device = input_ids.to(embed_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(embed_device)
        
        hidden_states = model.model.embed_tokens(input_ids_device)
        all_hidden_states.append(('embedding', hidden_states.clone()))
        
        outputs.append({
            'layer': 'embedding',
            'shape': tuple(hidden_states.shape),
            'mean': float(hidden_states.mean()),
            'std': float(hidden_states.std()),
            'min': float(hidden_states.min()),
            'max': float(hidden_states.max()),
            'has_nan': bool(torch.isnan(hidden_states).any()),
            'has_inf': bool(torch.isinf(hidden_states).any()),
        })
        
        # 逐层处理
        for layer_idx in range(min(num_layers, len(model.model.layers))):
            print(f"Processing layer {layer_idx}...", end='\r')
            layer = model.model.layers[layer_idx]
            
            # 确保hidden_states在正确的设备上
            layer_device = next(layer.parameters()).device
            hidden_states = hidden_states.to(layer_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(layer_device)
            
            try:
                # ===== Self Attention =====
                # attention_output, self_attention_weights, present_key_value
                attn_output = layer.self_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=False,
                )
                
                if isinstance(attn_output, tuple):
                    attn_output = attn_output[0]
                
                hidden_states = hidden_states + attn_output
                all_hidden_states.append((f'after_attn_{layer_idx}', hidden_states.clone()))
                
                # ===== Post Attention LayerNorm =====
                hidden_states = layer.post_attention_layernorm(hidden_states)
                all_hidden_states.append((f'after_post_attn_norm_{layer_idx}', hidden_states.clone()))
                
                # ===== MLP / MoE =====
                if model_type == "hf":
                    mlp_output = layer.mlp(hidden_states)
                    if isinstance(mlp_output, tuple):
                        mlp_output = mlp_output[0]
                else:
                    # 自定义模型可能有不同的输出格式
                    mlp_output = layer.mlp(hidden_states)
                    if isinstance(mlp_output, tuple):
                        mlp_output = mlp_output[0]
                
                hidden_states = hidden_states + mlp_output
                all_hidden_states.append((f'after_mlp_{layer_idx}', hidden_states.clone()))
                
                # ===== Input LayerNorm (用于下一层) =====
                hidden_states = layer.input_layernorm(hidden_states)
                all_hidden_states.append((f'after_input_norm_{layer_idx}', hidden_states.clone()))
                
                # 记录统计信息
                outputs.append({
                    'layer': f'layer_{layer_idx}',
                    'shape': tuple(hidden_states.shape),
                    'mean': float(hidden_states.mean()),
                    'std': float(hidden_states.std()),
                    'min': float(hidden_states.min()),
                    'max': float(hidden_states.max()),
                    'has_nan': bool(torch.isnan(hidden_states).any()),
                    'has_inf': bool(torch.isinf(hidden_states).any()),
                    'sample_values': hidden_states[0, 0, :10].cpu().tolist(),
                })
                
            except Exception as e:
                print(f"Error processing layer {layer_idx}: {e}")
                outputs.append({
                    'layer': f'layer_{layer_idx}',
                    'error': str(e),
                })
    
    print()  # 新行
    return outputs, (all_hidden_states if return_all_states else None)

def print_comparison(hf_outputs, custom_outputs, num_layers=5):
    """打印详细的对比信息"""
    print("\n" + "="*100)
    print(f"{'Layer':<30} {'HF Mean':<15} {'Custom Mean':<15} {'Diff':<15} {'Status':<10}")
    print("="*100)
    
    max_divergence = 0
    first_diverge_layer = None
    
    for i, (hf, custom) in enumerate(zip(hf_outputs, custom_outputs)):
        if 'error' in hf or 'error' in custom:
            print(f"{hf.get('layer', '?'):<30} {'ERROR':<15} {'ERROR':<15} {'N/A':<15}")
            continue
        
        layer_name = hf.get('layer', f'layer_{i}')
        hf_mean = hf.get('mean', float('nan'))
        hf_std = hf.get('std', float('nan'))
        custom_mean = custom.get('mean', float('nan'))
        custom_std = custom.get('std', float('nan'))
        
        diff = abs(hf_mean - custom_mean)
        max_divergence = max(max_divergence, diff)
        
        if diff > 0.1 and first_diverge_layer is None:
            first_diverge_layer = layer_name
        
        # 检测异常
        status = "✓"
        if hf.get('has_nan') or custom.get('has_nan'):
            status = "NaN !"
        elif hf.get('has_inf') or custom.get('has_inf'):
            status = "Inf !"
        elif diff > 1.0:
            status = "⚠️  HIGH"
        elif diff > 0.1:
            status = "⚠️  MED"
        
        print(f"{layer_name:<30} {hf_mean:<15.6f} {custom_mean:<15.6f} {diff:<15.6f} {status:<10}")
        
        # 打印详细的sample值
        if 'sample_values' in hf and 'sample_values' in custom:
            hf_sample = hf['sample_values'][:3]
            custom_sample = custom['sample_values'][:3]
            print(f"  HF Sample:     {[f'{v:.4f}' for v in hf_sample]}")
            print(f"  Custom Sample: {[f'{v:.4f}' for v in custom_sample]}")
    
    print("="*100)
    print(f"\nMax Divergence: {max_divergence:.6f}")
    if first_diverge_layer:
        print(f"First significant divergence at: {first_diverge_layer}")
    
    return max_divergence, first_diverge_layer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model', default='deepseek-ai/deepseek-moe-16b-base', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--text', default='The capital of France is', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    print("="*100)
    print("DEEPSEEK MODEL OUTPUT COMPARISON DEBUG")
    print("="*100)
    
    # 加载tokenizer
    print(f"\n[1/4] Loading tokenizer from {args.hf_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    
    # 准备输入
    print(f"[2/4] Preparing input: '{args.text}'")
    inputs = tokenizer(args.text, return_tensors='pt')
    # 当使用 device_map="auto" 时，获取模型的主设备
    model_device = next(hf_model.parameters()).device
    input_ids = inputs['input_ids'].to(model_device)
    attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(model_device)
    print(f"      Input shape: {input_ids.shape}")
    print(f"      Attention mask shape: {attention_mask.shape}")
    print(f"      Model device: {model_device}")
    
    # 加载HF模型
    print(f"[2/4] Loading HF model {args.hf_model}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分布到多张卡
        trust_remote_code=True
    )
    hf_model.eval()
    
    # 加载自定义模型
    print(f"[3/4] Loading custom model and copying weights...")
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)
    custom_config = DeepseekConfig(**config.to_dict())
    custom_model = DeepseekForCausalLM(custom_config).to(model_device).bfloat16()
    custom_model.eval()
    
    # 复制权重
    custom_model = copy_weights_to_custom_model(hf_model, custom_model, None, model_device)
    
    # 提取输出
    print(f"[4/4] Extracting layer outputs from both models...")
    print("\nHF Model:")
    hf_outputs, _ = extract_layer_outputs(
        hf_model, input_ids, attention_mask, 
        num_layers=args.num_layers, 
        model_type="hf"
    )
    
    print("Custom Model:")
    custom_outputs, _ = extract_layer_outputs(
        custom_model, input_ids, attention_mask, 
        num_layers=args.num_layers, 
        model_type="custom"
    )
    
    # 对比和打印
    max_div, first_div = print_comparison(hf_outputs, custom_outputs, args.num_layers)
    
    # 诊断建议
    print("\n" + "="*100)
    print("DIAGNOSTIC SUGGESTIONS")
    print("="*100)
    if max_div < 0.01:
        print("✓ Models are nearly identical - weights copied correctly")
    elif max_div < 0.1:
        print("⚠️  Minor divergence detected - may be due to dtype or RNG differences")
    elif max_div < 1.0:
        print("⚠️  Moderate divergence - check weight loading for the divergence layer")
        print(f"   First divergence at: {first_div}")
    else:
        print("❌ Major divergence - likely weight loading issue")
        print(f"   Check layer: {first_div}")
    
    print("\nChecklist:")
    print("  [ ] Verify weight dtype (bf16 vs fp32)")
    print("  [ ] Check layer normalization parameters")
    print("  [ ] Verify attention layer weights")
    print("  [ ] Check MLP/MoE weight loading")
    print("  [ ] Check for uninitialized parameters")
    print()

if __name__ == '__main__':
    main()
