"""
对比 HuggingFace Deepseek 模型和自定义模型的前5层输出
用于找出权重迁移中的问题
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.deepseek_moe import DeepseekForCausalLM, DeepseekConfig
import argparse
import os

def load_hf_model(model_name="deepseek-ai/deepseek-moe-16b-base", device="cuda:0"):
    """加载HuggingFace原始模型"""
    print(f"Loading HF model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    return model

def load_custom_model(model_path, device="cuda:0"):
    """加载自定义模型（已导入HF权重）"""
    print(f"Loading custom model from: {model_path}")
    # 假设模型已经通过某种方式初始化并导入了权重
    # 这里需要根据你的实际加载逻辑调整
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def get_intermediate_outputs(model, input_ids, attention_mask, num_layers=5, model_type="hf"):
    """获取模型前num_layers层的输出"""
    outputs = []
    
    if model_type == "hf":
        # HuggingFace模型
        with torch.no_grad():
            # 获取embedding层输出
            hidden_states = model.model.embed_tokens(input_ids)
            outputs.append({
                'layer': 'embedding',
                'shape': hidden_states.shape,
                'mean': hidden_states.mean().item(),
                'std': hidden_states.std().item(),
                'min': hidden_states.min().item(),
                'max': hidden_states.max().item(),
            })
            
            # 遍历前num_layers层
            for layer_idx in range(min(num_layers, len(model.model.layers))):
                layer = model.model.layers[layer_idx]
                
                # 通过attention层
                attention_output, _, _ = layer.self_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = hidden_states + attention_output
                
                # 通过norm
                hidden_states = layer.post_attention_layernorm(hidden_states)
                
                # 通过mlp
                mlp_output = layer.mlp(hidden_states)
                hidden_states = hidden_states + mlp_output
                
                # 通过norm
                hidden_states = layer.input_layernorm(hidden_states)
                
                outputs.append({
                    'layer': f'layer_{layer_idx}',
                    'shape': hidden_states.shape,
                    'mean': hidden_states.mean().item(),
                    'std': hidden_states.std().item(),
                    'min': hidden_states.min().item(),
                    'max': hidden_states.max().item(),
                    'sample': hidden_states[0, 0, :5].cpu().tolist(),  # 第一个token的前5个值
                })
    
    else:
        # 自定义模型
        with torch.no_grad():
            hidden_states = model.model.embed_tokens(input_ids)
            outputs.append({
                'layer': 'embedding',
                'shape': hidden_states.shape,
                'mean': hidden_states.mean().item(),
                'std': hidden_states.std().item(),
                'min': hidden_states.min().item(),
                'max': hidden_states.max().item(),
            })
            
            for layer_idx in range(min(num_layers, len(model.model.layers))):
                layer = model.model.layers[layer_idx]
                
                # 通过attention层
                attention_output, _, _ = layer.self_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = hidden_states + attention_output
                
                # 通过norm
                hidden_states = layer.post_attention_layernorm(hidden_states)
                
                # 通过mlp
                mlp_output = layer.mlp(hidden_states)
                if isinstance(mlp_output, tuple):
                    mlp_output = mlp_output[0]
                hidden_states = hidden_states + mlp_output
                
                # 通过norm
                hidden_states = layer.input_layernorm(hidden_states)
                
                outputs.append({
                    'layer': f'layer_{layer_idx}',
                    'shape': hidden_states.shape,
                    'mean': hidden_states.mean().item(),
                    'std': hidden_states.std().item(),
                    'min': hidden_states.min().item(),
                    'max': hidden_states.max().item(),
                    'sample': hidden_states[0, 0, :5].cpu().tolist(),
                })
    
    return outputs

def compare_outputs(hf_outputs, custom_outputs):
    """对比两个模型的输出"""
    print("\n" + "="*80)
    print("OUTPUT COMPARISON (HuggingFace vs Custom Model)")
    print("="*80)
    
    for i, (hf_out, custom_out) in enumerate(zip(hf_outputs, custom_outputs)):
        layer_name = hf_out['layer']
        print(f"\n{layer_name.upper()}")
        print("-" * 60)
        
        # 比较形状
        hf_shape = hf_out['shape']
        custom_shape = custom_out['shape']
        shape_match = "✓" if hf_shape == custom_shape else "✗"
        print(f"Shape {shape_match}: HF={hf_shape}, Custom={custom_shape}")
        
        # 比较统计信息
        print(f"Mean:  HF={hf_out['mean']:.6f}, Custom={custom_out['mean']:.6f}, Diff={abs(hf_out['mean']-custom_out['mean']):.6f}")
        print(f"Std:   HF={hf_out['std']:.6f}, Custom={custom_out['std']:.6f}, Diff={abs(hf_out['std']-custom_out['std']):.6f}")
        print(f"Min:   HF={hf_out['min']:.6f}, Custom={custom_out['min']:.6f}")
        print(f"Max:   HF={hf_out['max']:.6f}, Custom={custom_out['max']:.6f}")
        
        if 'sample' in hf_out and 'sample' in custom_out:
            print(f"Sample values (first 5):")
            print(f"  HF:     {[f'{v:.4f}' for v in hf_out['sample']]}")
            print(f"  Custom: {[f'{v:.4f}' for v in custom_out['sample']]}")
        
        # 检测异常
        if abs(hf_out['mean'] - custom_out['mean']) > 1.0:
            print("⚠️  MODEL DIVERGENCE DETECTED - Mean difference > 1.0")
        if float('nan') in str([hf_out['mean'], hf_out['std'], custom_out['mean'], custom_out['std']]):
            print("⚠️  NaN DETECTED - Possible numerical issue")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-ai/deepseek-moe-16b-base', type=str)
    parser.add_argument('--custom_model_path', required=True, type=str, help='Path to custom model checkpoint')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--text', default='The capital of France is', type=str)
    args = parser.parse_args()
    
    device = args.device
    torch.cuda.set_device(device)
    
    # 加载tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 准备输入
    print(f"Preparing input: '{args.text}'")
    inputs = tokenizer(args.text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device) if 'attention_mask' in inputs else None
    print(f"Input shape: {input_ids.shape}")
    
    # 加载模型
    hf_model = load_hf_model(args.model_name, device)
    
    # 这里需要根据你的实际情况调整
    # 可能需要从checkpoint加载，或者使用特定的加载方法
    try:
        custom_model = load_custom_model(args.custom_model_path, device)
    except Exception as e:
        print(f"Could not load custom model directly, trying alternative method...")
        print(f"Error: {e}")
        print("\nPlease update the load_custom_model() function to match your loading logic")
        return
    
    # 获取输出
    print(f"\nExtracting outputs from both models...")
    hf_outputs = get_intermediate_outputs(hf_model, input_ids, attention_mask, 
                                          num_layers=args.num_layers, model_type="hf")
    custom_outputs = get_intermediate_outputs(custom_model, input_ids, attention_mask, 
                                              num_layers=args.num_layers, model_type="custom")
    
    # 对比输出
    compare_outputs(hf_outputs, custom_outputs)
    
    print("\n" + "="*80)
    print("ANALYSIS TIPS:")
    print("="*80)
    print("""
1. If embedding layer outputs match:
   - Problem is likely in the transformer layers
   
2. If first layer diverges significantly:
   - Check attention layer weight transfer
   - Check MLP layer weights
   - Verify layer normalization parameters
   
3. If outputs are NaN:
   - Check for dtype mismatches (bf16 vs fp32)
   - Check for uninitialized parameters
   
4. If outputs diverge gradually:
   - Problem is likely accumulating in each layer
   - Check residual connections or layer norm scaling
   """)

if __name__ == '__main__':
    main()
