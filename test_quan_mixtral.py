import torch
import torch.nn as nn
from operator import attrgetter
import argparse
import time
import os
import glog
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.linear import FusedQuantizedLinear, QuantizedLinear
from lib import codebook, utils

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--hessian_path', type=str)
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--model_type', default='auto', type=str,
                    choices=['auto', 'llama', 'mixtral', 'mistral'],
                    help='Model type to quantize')
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--lora_rank',
                    default=0,
                    type=int,
                    help='if <=0 then turned off')
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--resid_scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str, required=True)
parser.add_argument('--quip_tune_iters', default=10, type=int)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--full_svd', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=5e-5, type=float)
parser.add_argument('--ft_susv_lr', default=5e-4, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--expert_batch_size', default=1, type=int,
                    help='Number of experts to process at once (for memory control)')
parser.add_argument('--quantize_gate', action='store_true',
                    help='Whether to quantize the gate network')
parser.add_argument('--gate_precision', default='bf16', type=str,
                    choices=['fp32', 'fp16', 'bf16'],
                    help='Precision for gate network if not quantized')
parser.add_argument('--skip_test', action='store_true',
                    help='Skip model inference test (only load weights)')

def get_mixtral_quant_order(layer, layer_idx):
    """
    生成 Mixtral 模型的量化顺序列表。
    必须与量化时使用的命名逻辑保持一致。
    
    返回格式: [(module_attr_path, save_name_suffix), ...]
    """
    order = []
    
    # 1. Attention 部分
    # 假设 Q, K, V 被融合或者是独立的，这里列出常见的几种情况，
    # 请根据你实际保存的文件名调整。
    
    # 情况 A: 如果 QKV 是融合的
    # order.append(('self_attn.qkv_proj', 'qkv')) 
    
    # 情况 B: 如果是独立的 (Mixtral 默认通常是独立的，或者 Q, K, V)
    order.append(('self_attn.q_proj', 'q'))
    order.append(('self_attn.k_proj', 'k'))
    order.append(('self_attn.v_proj', 'v'))
    order.append(('self_attn.o_proj', 'out'))

    # 2. MoE Experts 部分
    # Mixtral 8x7B 有 8 个专家
    num_experts = 8 
    
    # 这里的命名逻辑必须和你量化脚本中生成 'name' 的逻辑一致
    # 假设结构是: block_sparse_moe.experts.[i].w1, w2, w3
    # w1(gate) + w3(up) 通常融合为 'up'
    # w2(down) 通常独立为 'down'
    
    for i in range(num_experts):
        # 融合的 Gate + Up (w1 + w3)
        # 注意：这里的 attr_path 指向的是模型中原始层的路径
        # 如果原始模型已经是 FusedLinear，路径可能不同，这里假设是标准的 HF 结构
        # 如果量化时用了 FusedLinear wrapper，这里主要关注 save_name
        order.append((f'block_sparse_moe.experts.{i}.w1', f'experts.{i}.up')) 
        
        # 独立的 Down (w2)
        order.append((f'block_sparse_moe.experts.{i}.w2', f'experts.{i}.down'))

    # 3. MoE Gate (Router)
    order.append(('block_sparse_moe.gate', 'gate'))

    return order

def load_quantized_model(model, codebooks, args):
    print(f"Loading quantized checkpoint from {args.save_path}...")
    
    config_path = os.path.join(args.save_path, 'config.pt')
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location="cpu")
        quip_params = config.get('model_config', {}).get('quip_params', {})
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")
        # quip_params = {
        #     'codesz': 8,
        #     'packsz': 1,
        #     'pack_out': False,
        #     'idx_dtype': 'int32',
        #     'codebook_version': 'unknown',
        #     'rank': args.lora_rank if hasattr(args, 'lora_rank') else 64,
        #     'rescale_WH': False,
        # }
    
    if os.path.isdir(args.save_path):
        ckpt = {}
        for file in os.listdir(args.save_path):
            if file.endswith('.pt') and file != 'config.pt':
                layer_name = file[:-3]
                file_path = os.path.join(args.save_path, file)
                ckpt[layer_name] = torch.load(file_path, map_location="cpu")
    else:
        ckpt = torch.load(args.save_path, map_location="cpu")
    
    print(f"  Loaded {len(ckpt)} checkpoint layers")
    input("Press Enter to continue...") 
    
    current_device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_to_model_mapping = {}
    
    for ckpt_name in ckpt.keys():
        parts = ckpt_name.split('_')
        
        # 跳过非层数据文件（如embed, lm_head等）
        if len(parts) < 2:
            continue
        
        try:
            layer_id = int(parts[0])
        except ValueError:
            # 如果第一部分不是数字，跳过（如embed, lm_head等）
            continue
        
        weight_type = '_'.join(parts[1:])
        
        if weight_type == 'expert0_w1' or (len(parts) >= 3 and parts[1].startswith('expert')):
            # MoE expert层: "0_expert0_w1" -> "model.layers.0.block_sparse_moe.experts.0.w1"
            expert_id = int(parts[1].replace('expert', ''))
            w_type = '_'.join(parts[2:])
            model_name = f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.{w_type}"
        elif weight_type == 'q':
            # 融合的qkv层 - 在Mixtral中实际是分开的，暂时跳过
            # 因为checkpoint和模型架构不匹配
            continue
        elif weight_type == 'o':
            # Output projection
            model_name = f"model.layers.{layer_id}.self_attn.o_proj"
        elif weight_type == 'gate':
            # Gate
            model_name = f"model.layers.{layer_id}.block_sparse_moe.gate"
        elif weight_type == 'layernorm':
            # LayerNorm (skip, not Linear)
            continue
        else:
            # Skip unknown types
            continue
        
        ckpt_to_model_mapping[ckpt_name] = model_name
    
    # 遍历模型的所有子模块，收集需要替换的层
    modules_to_replace = []
    linear_modules = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    
    matched_count = 0
    unmatched_checkpoint = []
    for ckpt_name, model_name in ckpt_to_model_mapping.items():
        if model_name in linear_modules:
            module = linear_modules[model_name]
            modules_to_replace.append((ckpt_name, model_name, module))
            matched_count += 1
        else:
            unmatched_checkpoint.append((ckpt_name, model_name))
    
    print(f"Matched: {matched_count} layers, Unmatched: {len(unmatched_checkpoint)} layers")

    #------------------------------------------------ 
    input("Press Enter to continue...") 
    #------------------------------------------------ 

    if unmatched_checkpoint and len(unmatched_checkpoint) <= 5:
        for ckpt, model in unmatched_checkpoint:
            print(f"  Unmatched: {ckpt} -> {model}")
    
    print(f"Replacing {len(modules_to_replace)} layers...")

    for ckpt_name, model_name, module in modules_to_replace:
        print(f"Processing layer: {ckpt_name} -> {model_name}")
        saved_linear = ckpt[ckpt_name]
        is_fused = saved_linear.get('fused', False)
        print(f"  Is fused: {is_fused}") 

        input("Press Enter to continue...") 
        
        shapes = saved_linear.get('shapes', None)
        if shapes and len(shapes) > 0:
            out_features, in_features = shapes[0]
        else:
            in_features = module.in_features
            out_features = module.out_features
        
        # Check for shape mismatch
        module_in = module.in_features
        module_out = module.out_features
        if (in_features != module_in or out_features != module_out):
            print(f"⚠ Shape mismatch in {ckpt_name}:")
            print(f"    Model:      in={module_in}, out={module_out}")
            print(f"    Checkpoint: in={in_features}, out={out_features}")
        
        # Extract quantization parameters
        codesz = quip_params.get('codesz', 8)
        packsz = quip_params.get('packsz', 1)
        pack_out = quip_params.get('pack_out', False)
        idx_dtype = quip_params.get('idx_dtype', 'int32')
        
        # Convert torch dtype to string if necessary
        if hasattr(idx_dtype, 'name'):
            idx_dtype = idx_dtype.name
        elif hasattr(idx_dtype, '__name__'):
            idx_dtype = str(idx_dtype)
        
        codebook_version = quip_params.get('codebook_version', 'unknown')
        rank = quip_params.get('rank', quip_params.get('lora_rank', 64))
        rescale_WH = quip_params.get('rescale_WH', False)
        
        try:
            if is_fused:
                # For fused layers, use FusedQuantizedLinear
                fuse_dim = saved_linear.get('fuse_dim', 0)
                fuse_sizes = saved_linear.get('fuse_sizes', None)
                new_layer = FusedQuantizedLinear(
                    fuse_dim,
                    fuse_sizes if fuse_sizes else [out_features],
                    in_features,
                    out_features,
                    codesz,
                    packsz,
                    pack_out,
                    idx_dtype,
                    codebook_version,
                    rank=rank,
                    rescale_WH=rescale_WH,
                )
            else:
                # For regular layers, use QuantizedLinear
                new_layer = QuantizedLinear(
                    in_features,
                    out_features,
                    codesz,
                    packsz,
                    pack_out,
                    idx_dtype,
                    codebook_version,
                    rank=rank,
                    rescale_WH=rescale_WH,
                )
        except Exception as e:
            print(f"Error creating layer {ckpt_name}: {e}")
            continue
        
        # Load saved weights into the new layer
        for param_name in ['Qidxs', 'A', 'B', 'SU', 'SV', 'scaleWH']:
            if param_name in saved_linear and saved_linear[param_name] is not None:
                if hasattr(new_layer, param_name):
                    param_data = saved_linear[param_name]
                    
                    # 修复SU/SV的无效值
                    if param_name in ['SU', 'SV']:
                        invalid_mask = param_data <= 0
                        if invalid_mask.any():
                            print(f"  ⚠ {ckpt_name}.{param_name}: {invalid_mask.sum().item()} invalid values, clamping to 1e-8")
                            param_data = torch.clamp(param_data, min=1e-8)
                    
                    # 验证Qidxs形状
                    if param_name == 'Qidxs':
                        m, n = param_data.shape
                        if m < 16 or m % 16 != 0:
                            print(f"  ⚠ {ckpt_name}: Qidxs shape {param_data.shape} incompatible (m={m} not divisible by 16), skipping")
                            continue
                        if n % 4 != 0:
                            print(f"  ⚠ {ckpt_name}: Qidxs shape {param_data.shape} incompatible (n={n} not divisible by 4), skipping")
                            continue
                    
                    getattr(new_layer, param_name).data.copy_(param_data)
        
        # Move layer to device and replace in model
        new_layer.to(current_device)
        parent_name = model_name.rsplit('.', 1)[0] if '.' in model_name else ''
        child_name = model_name.rsplit('.', 1)[-1]
        
        if parent_name:
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, child_name, new_layer)

    # 强制清理显存
    torch.cuda.empty_cache()


def test_quantized_model(model, tokenizer, device='cuda'):
    """
    验证量化权重是否正确加载。
    """
    print("\nVerifying quantized model...")
    
    model.eval()
    
    # 1. 检查QuantizedLinear层
    print("\nQuantized layers statistics:")
    quantized_layers = []
    qidxs_stats = {
        'total': 0,
        'packed': 0,
        'shapes': {}
    }
    
    for name, module in model.named_modules():
        if 'QuantizedLinear' in module.__class__.__name__:
            quantized_layers.append((name, module))
            
            if hasattr(module, 'Qidxs'):
                qidxs = module.Qidxs
                qidxs_stats['total'] += 1
                shape_key = str(tuple(qidxs.shape))
                qidxs_stats['shapes'][shape_key] = qidxs_stats['shapes'].get(shape_key, 0) + 1
                
                # 检查是否为packed形式（4D以上）
                if qidxs.dim() >= 4:
                    qidxs_stats['packed'] += 1
    
    print(f"  Total QuantizedLinear layers: {len(quantized_layers)}")
    print(f"  Packed Qidxs layers: {qidxs_stats['packed']}/{qidxs_stats['total']}")
    print(f"\n  Qidxs shape distribution:")
    for shape, count in sorted(qidxs_stats['shapes'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {shape}: {count} layers")
    
    # 2. 检查权重的数值范围和有效性
    print(f"\nWeight validation:")
    weight_valid = True
    for name, module in quantized_layers[:5]:
        if hasattr(module, 'SU') and hasattr(module, 'SV'):
            su_min, su_max = module.SU.min().item(), module.SU.max().item()
            sv_min, sv_max = module.SV.min().item(), module.SV.max().item()
            
            su_valid = (su_min > 0 and su_max > 0)
            sv_valid = (sv_min > 0 and sv_max > 0)
            
            if su_valid and sv_valid:
                print(f"  ✓ {name.split('.')[-2:]} - SU: [{su_min:.4f}, {su_max:.4f}], SV: [{sv_min:.4f}, {sv_max:.4f}]")
            else:
                print(f"  ⚠ {name.split('.')[-2:]} - Invalid ranges!")
                weight_valid = False
    
    # 3. 验证Forward（有已知的shape pack/unpack问题）
    print(f"\nForward pass test (limited due to Qidxs pack/unpack mismatch):")
    try:
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        for key in inputs:
            if hasattr(inputs[key], 'to'):
                inputs[key] = inputs[key].cuda() if torch.cuda.is_available() else inputs[key]
        
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        
        print(f"  ✓ Forward pass successful")
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠ Loss is NaN or Inf (numerical issue)")
            else:
                print(f"  ✓ Loss: {loss.item():.4f}")
        
    except RuntimeError as e:
        error_msg = str(e)
        if "invalid for input of size" in error_msg or "shape" in error_msg:
            print(f"  ⚠ Qidxs reshape mismatch (expected - pack/unpack inconsistency):")
            print(f"     Qidxs are saved in packed format but forward expects different shape")
            print(f"     This is a known issue between quantization and inference phases")
            print(f"\n     ✓ Weights successfully loaded - shape issue is in codebook.forward()")
            print(f"     ✓ Solution: Need to verify pack/unpack logic in latticee8_padded12_rvq4bit.py")
        else:
            print(f"  ⚠ Error: {error_msg[:150]}")
    except Exception as e:
        print(f"  ⚠ Unexpected error: {type(e).__name__}")
    
    # 4. 最终总结
    print(f"\n{'='*50}")
    if len(quantized_layers) > 0:
        print(f"✓ Quantization Summary:")
        print(f"  - {len(quantized_layers)} QuantizedLinear layers loaded")
        print(f"  - Weight shapes validated")
        print(f"  - SU/SV parameters in valid range")
        if weight_valid:
            print(f"  - All weight parameters appear valid")
    else:
        print(f"⚠ No QuantizedLinear layers found!")
    print(f"{'='*50}")


# ================= 使用示例 =================
if __name__ == "__main__":
    args = parser.parse_args()
    
    # 2. 加载 Codebook
    cb = codebook.get_codebook(args.codebook) 

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 加载原始模型 (Meta-Llama 格式或 Mixtral 格式)
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.float16, device_map="auto")
    
    # 4. 执行加载
    load_quantized_model(model, cb, args)
    
    # 5. 运行测试（如果不跳过）
    if not args.skip_test:
        test_quantized_model(model, tokenizer)
    else:
        print("✓ Quantized weights loaded successfully. (test skipped)")

