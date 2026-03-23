"""
调试脚本：验证quantize_finetune_deepseek_s.py中的权重复制是否正确
"""
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from transformers import AutoModelForCausalLM, AutoConfig
from model.deepseek_moe import DeepseekForCausalLM, DeepseekConfig
import copy

def compare_weights(layer_hf, layer_custom, layer_idx):
    """比较HF层和自定义层的权重是否一致"""
    print(f"\n{'='*80}")
    print(f"Layer {layer_idx} Weight Comparison")
    print(f"{'='*80}")
    
    # 比较attention权重
    attn_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    for mod in attn_modules:
        hf_weight = getattr(layer_hf.self_attn, mod).weight
        custom_weight = getattr(layer_custom.self_attn, mod).weight
        
        if hf_weight.shape != custom_weight.shape:
            print(f"❌ {mod}: Shape mismatch! HF: {hf_weight.shape}, Custom: {custom_weight.shape}")
        else:
            diff = (hf_weight - custom_weight).abs().max().item()
            print(f"   {mod}: Max diff = {diff:.2e}")
            if diff > 1e-5:
                print(f"      ⚠️  WARNING: Large difference detected!")
    
    # 比较LayerNorm权重
    if hasattr(layer_hf, 'input_layernorm'):
        hf_ln = layer_hf.input_layernorm.weight
        custom_ln = layer_custom.input_layernorm.weight
        diff = (hf_ln - custom_ln).abs().max().item()
        print(f"   input_layernorm: Max diff = {diff:.2e}")
        if diff > 1e-5:
            print(f"      ⚠️  WARNING: Large difference detected!")
    
    # 比较MLP权重（如果是Dense MLP）
    if hasattr(layer_hf.mlp, 'gate_proj') and not hasattr(layer_hf.mlp, 'experts'):
        print(f"   Dense MLP detected")
        mlp_modules = ['gate_proj', 'up_proj', 'down_proj']
        for mod in mlp_modules:
            hf_weight = getattr(layer_hf.mlp, mod).weight
            custom_weight = getattr(layer_custom.mlp, mod).weight
            
            if hf_weight.shape != custom_weight.shape:
                print(f"   ❌ {mod}: Shape mismatch! HF: {hf_weight.shape}, Custom: {custom_weight.shape}")
            else:
                diff = (hf_weight - custom_weight).abs().max().item()
                print(f"   {mod}: Max diff = {diff:.2e}")
                if diff > 1e-5:
                    print(f"      ⚠️  WARNING: Large difference detected!")
    
    # 比较MoE权重（如果是MoE）
    if hasattr(layer_hf.mlp, 'gate'):
        print(f"   MoE layer detected (with gate)")
        hf_gate = layer_hf.mlp.gate.weight
        custom_gate = layer_custom.mlp.gate.weight
        if hf_gate.shape != custom_gate.shape:
            print(f"   ❌ gate: Shape mismatch! HF: {hf_gate.shape}, Custom: {custom_gate.shape}")
        else:
            diff = (hf_gate - custom_gate).abs().max().item()
            print(f"   gate: Max diff = {diff:.2e}")


def main():
    model_name = 'deepseek-ai/deepseek-moe-16b-base'
    
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    hf_model.eval()
    
    print("Creating custom model...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    custom_config = DeepseekConfig(**config.to_dict())
    custom_model = DeepseekForCausalLM(custom_config)
    
    # 方法1：使用deepcopy替换（quantize_finetune_deepseek_s.py的方式）
    print("\nMethod 1: Using deepcopy (current quantize_finetune script method)")
    custom_model_deepcopy = DeepseekForCausalLM(custom_config)
    
    for idx in range(min(2, len(hf_model.model.layers))):  # 只测试前2层
        hf_layer = hf_model.model.layers[idx]
        custom_layer = custom_model_deepcopy.model.layers[idx]
        
        # 用deepcopy替换（当前方式）
        custom_layer.self_attn.q_proj = copy.deepcopy(hf_layer.self_attn.q_proj)
        custom_layer.self_attn.k_proj = copy.deepcopy(hf_layer.self_attn.k_proj)
        custom_layer.self_attn.v_proj = copy.deepcopy(hf_layer.self_attn.v_proj)
        custom_layer.self_attn.o_proj = copy.deepcopy(hf_layer.self_attn.o_proj)
        
        custom_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        custom_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
        
        if hasattr(hf_layer.mlp, 'gate_proj') and not hasattr(hf_layer.mlp, 'experts'):
            custom_layer.mlp.gate_proj = copy.deepcopy(hf_layer.mlp.gate_proj)
            custom_layer.mlp.up_proj = copy.deepcopy(hf_layer.mlp.up_proj)
            custom_layer.mlp.down_proj = copy.deepcopy(hf_layer.mlp.down_proj)
        
        compare_weights(hf_layer, custom_layer, idx)
    
    # 方法2：使用load_state_dict（推荐方式）
    print("\n\nMethod 2: Using load_state_dict (recommended)")
    custom_model_statedict = DeepseekForCausalLM(custom_config)
    
    # 复制HF模型的state_dict到自定义模型
    hf_state = hf_model.state_dict()
    custom_model_statedict.load_state_dict(hf_state, strict=False)
    
    for idx in range(min(2, len(hf_model.model.layers))):
        hf_layer = hf_model.model.layers[idx]
        custom_layer = custom_model_statedict.model.layers[idx]
        compare_weights(hf_layer, custom_layer, idx)
    
    print("\n" + "="*80)
    print("Summary: 权重应该完全一致（diff ≈ 0）")
    print("="*80)


if __name__ == "__main__":
    main()
