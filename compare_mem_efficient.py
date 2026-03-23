"""
超节省显存的对比脚本 - 比较 HuggingFace 和自定义模型的前5层输出
支持多GPU自动分布，处理显存不足问题
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import dispatch_model, infer_auto_device_map
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
    print("DEEPSEEK MODEL COMPARISON - Ultra Memory Efficient")
    print("="*100)
    
    # 加载tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 准备输入
    print(f"[2/4] Preparing input: '{args.text}'")
    inputs = tokenizer(args.text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
    print(f"      Input shape: {input_ids.shape}")
    
    # ===== STEP 1: 提取 HF 模型的输出 =====
    print(f"\n[3/4a] Loading and processing HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分布到所有可用GPU
        trust_remote_code=True
    )
    hf_model.eval()
    
    main_device = next(hf_model.parameters()).device
    input_ids_device = input_ids.to(main_device)
    attention_mask_device = attention_mask.to(main_device)
    
    # 为 attention 层构造正确的掩码维度
    # 需要完整的因果mask：(batch_size, 1, seq_len, seq_len)
    seq_len = input_ids_device.shape[1]
    
    # 创建因果mask (seq_len, seq_len)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=main_device), diagonal=1).bool()
    # 扩展到 (batch_size, 1, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    # 转换为注意力形式：masked位置为min_float，使用bfloat16匹配模型dtype
    attention_mask_4d = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16, device=main_device)
    attention_mask_4d.masked_fill_(causal_mask, torch.finfo(torch.bfloat16).min)
    
    position_ids = torch.arange(seq_len, dtype=torch.long, device=main_device).unsqueeze(0)
    
    print(f"      HF model on: {main_device}")
    print(f"      Input shape: {input_ids_device.shape}")
    print(f"      Position IDs shape: {position_ids.shape}")
    print(f"      4D attention mask shape: {attention_mask_4d.shape}")
    
    hf_outputs = []
    
    with torch.no_grad():
        # Embedding 层
        hf_hidden = hf_model.model.embed_tokens(input_ids_device)
        hf_outputs.append(print_layer_stats('embedding', hf_hidden))
        
        # 逐层处理
        for layer_idx in range(min(args.num_layers, len(hf_model.model.layers))):
            print(f"      HF Layer {layer_idx}...", end='\r')
            layer = hf_model.model.layers[layer_idx]
            
            # Self Attention + residual
            residual = hf_hidden
            hf_hidden = layer.input_layernorm(hf_hidden)
            attn_out = layer.self_attn(hf_hidden, attention_mask=attention_mask_4d, position_ids=position_ids)[0]
            hf_hidden = residual + attn_out
            
            # MLP + residual
            residual = hf_hidden
            hf_hidden = layer.post_attention_layernorm(hf_hidden)
            mlp_out = layer.mlp(hf_hidden)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            hf_hidden = residual + mlp_out
            
            hf_outputs.append(print_layer_stats(f'layer_{layer_idx}', hf_hidden))
    
    print("\n      Deleting HF model...")
    del hf_model, input_ids_device, attention_mask_device, hf_hidden
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()  # 清空两次确保显存释放
    gc.collect()
    
    # ===== STEP 2: 创建和加载自定义模型 =====
    print(f"\n[3/4b] Creating and loading custom model...")
    
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    custom_config = DeepseekConfig(**config.to_dict())
    
    # 直接用device_map加载自定义模型的权重到分布式设备
    print("      Loading custom model with auto distribution across GPUs...")
    
    # 先用HF模型作为权重源，直接用device_map加载
    hf_model_temp = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 从HF模型提取state dict并复制到自定义模型各层
    hf_state_dict = hf_model_temp.state_dict()
    
    # 删除临时模型并释放显存
    del hf_model_temp
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    
    # 在CPU上创建自定义模型
    print("      Creating custom model structure...")
    custom_model = DeepseekForCausalLM(custom_config)
    custom_model.load_state_dict(hf_state_dict, strict=False)
    del hf_state_dict
    gc.collect()
    
    # 现在用device_map分布到多GPU
    print("      Distributing across GPUs with device_map...")
    from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map
    
    # 获取HF的设备映射配置
    max_memory = {i: f"40GB" for i in range(torch.cuda.device_count())}
    device_map = infer_auto_device_map(
        custom_model,
        max_memory=max_memory,
        no_split_module_classes=["DeepseekDecoderLayer"]
    )
    
    custom_model = dispatch_model(
        custom_model.to(torch.device('cpu')), 
        device_map=device_map
    )
    custom_model.eval()
    
    # ===== STEP 3: 提取自定义模型的输出 =====
    print(f"\n[4/4] Processing custom model...")
    
    # 获取embedding层所在的设备（模型现在分布在多卡上）
    embed_device = next(custom_model.model.embed_tokens.parameters()).device
    input_ids_device = input_ids.to(embed_device)
    
    # 构造正确的掩码维度 - 完整的因果mask
    seq_len = input_ids_device.shape[1]
    
    # 创建因果mask (seq_len, seq_len)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=embed_device), diagonal=1).bool()
    # 扩展到 (batch_size, 1, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    # 转换为注意力形式：masked位置为min_float，使用bfloat16匹配模型dtype
    attention_mask_4d = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16, device=embed_device)
    attention_mask_4d.masked_fill_(causal_mask, torch.finfo(torch.bfloat16).min)
    
    position_ids = torch.arange(seq_len, dtype=torch.long, device=embed_device).unsqueeze(0)
    
    custom_outputs = []
    
    with torch.no_grad():
        # Embedding 层
        custom_hidden = custom_model.model.embed_tokens(input_ids_device)
        custom_outputs.append(print_layer_stats('embedding', custom_hidden))
        
        # 逐层处理
        for layer_idx in range(min(args.num_layers, len(custom_model.model.layers))):
            print(f"      Custom Layer {layer_idx}...", end='\r')
            layer = custom_model.model.layers[layer_idx]
            
            # 确保在正确设备
            layer_device = next(layer.parameters()).device
            custom_hidden = custom_hidden.to(layer_device)
            attention_mask_lay = attention_mask_4d.to(layer_device)
            position_ids_lay = position_ids.to(layer_device)
            
            # Self Attention + residual
            residual = custom_hidden
            custom_hidden = layer.input_layernorm(custom_hidden)
            attn_out = layer.self_attn(custom_hidden, attention_mask=attention_mask_lay, position_ids=position_ids_lay)[0]
            custom_hidden = residual + attn_out
            
            # MLP + residual
            residual = custom_hidden
            custom_hidden = layer.post_attention_layernorm(custom_hidden)
            mlp_out = layer.mlp(custom_hidden)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            custom_hidden = residual + mlp_out
            
            custom_outputs.append(print_layer_stats(f'layer_{layer_idx}', custom_hidden))
    
    print("\n")
    
    # ===== 打印对比结果 =====
    print("="*100)
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
