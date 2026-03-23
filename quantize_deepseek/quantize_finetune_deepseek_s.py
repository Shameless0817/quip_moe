import argparse
import os
import sys
import time
import json
from tqdm import tqdm
import copy
from operator import attrgetter

# 添加父目录到Python路径，以便能够导入lib模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from model.deepseek_moe import DeepseekForCausalLM, DeepseekConfig

from lib import codebook, utils
from lib.algo import quip
from lib.linear import FusedLinear

from lib.algo import finetune_deepseek

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--hessian_path', type=str, help='Legacy: single Hessian path for all layers')
parser.add_argument('--dense_hessian_path', type=str, help='Hessian path for dense layers (attention Q/K/V/O)')
parser.add_argument('--sparse_hessian_path', type=str, help='Hessian path for sparse layers (expert FFN)')
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--model_type', default='auto', type=str,
                    choices=['auto', 'llama', 'mixtral', 'mistral', 'deepseek'],
                    help='Model type to quantize')
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--had_block_size',
                    default=64,
                    type=int,
                    help='Block size for blockwise Hadamard transform (must be power of 2). Used when dimension is incompatible with standard Hadamard.')
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
parser.add_argument('--skip_generation_test', action='store_true', default=False,
                    help='Skip generation test after quantization (saves GPU memory). Default is True.')
parser.add_argument('--run_generation_test', action='store_true', 
                    help='Run generation test after quantization (may cause OOM). Disable --skip_generation_test if using this.')
parser.add_argument('--compare_linear_outputs', action='store_true',
                    help='Compare original vs quantized linear outputs during quantization.')
parser.add_argument('--compare_max_tokens', default=2048, type=int,
                    help='Max number of token activations sampled per linear for output diff comparison.')
parser.add_argument('--compare_seq_len', default=128, type=int,
                    help='Sequence length used when collecting layer activations for output diff comparison.')
parser.add_argument('--compare_rel_rmse_warn', default=0.10, type=float,
                    help='Warn when relative RMSE exceeds this threshold.')
parser.add_argument('--compare_cosine_warn', default=0.995, type=float,
                    help='Warn when cosine similarity is below this threshold.')


def check_exist(idx, layer, args):
    """检查 DeepSeek 层的量化文件是否已存在"""
    quant_order = finetune_deepseek.get_deepseek_quant_order(layer)
    
    # 检查所有线性层是否已经量化
    for _, name in quant_order:
        if not os.path.exists(f'{args.save_path}/{idx}_{name}.pt'):
            return False
            
    # 检查 gate 和 layernorm
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate') and not args.quantize_gate:
        if not os.path.exists(f'{args.save_path}/{idx}_gate.pt'):
            return False
            
    if not os.path.exists(f'{args.save_path}/{idx}_layernorm.pt'):
        return False
        
    return True


def _collect_linear_inputs_for_compare(layer, quant_order, layer_input_emb, args, device):
    """Collect representative inputs for each target linear in quant_order via forward pre-hooks."""
    if layer_input_emb is None or layer_input_emb.numel() == 0:
        return {}

    sample_bs = max(1, min(layer_input_emb.shape[0], 2))
    sample_seq = max(1, min(layer_input_emb.shape[1], args.compare_seq_len))
    sample_hidden = layer_input_emb[:sample_bs, :sample_seq].contiguous()

    if str(device).startswith('cuda') and torch.cuda.is_available():
        sample_hidden = sample_hidden.to(device)
    else:
        sample_hidden = sample_hidden.cpu()

    position_ids = torch.arange(sample_seq, dtype=torch.int32, device=sample_hidden.device)[None, :]
    position_ids = position_ids + torch.zeros(sample_bs, sample_seq, dtype=torch.int32, device=sample_hidden.device)
    attention_mask = _prepare_4d_causal_attention_mask(
        None,
        (sample_bs, sample_seq),
        sample_hidden,
        0,
    )

    captured = {}
    hooks = []

    def make_hook(name):
        def _hook(module, inputs):
            if name in captured or len(inputs) == 0:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.detach().float().reshape(-1, x.shape[-1]).cpu()
            if x.shape[0] > args.compare_max_tokens:
                x = x[:args.compare_max_tokens]
            captured[name] = x
        return _hook

    for linear_attr, name in quant_order:
        try:
            module = attrgetter(linear_attr)(layer)
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
        except Exception as e:
            glog.warning(f'Failed to register hook for {name} ({linear_attr}): {e}')

    prev_mode = layer.training
    layer.eval()
    with torch.no_grad():
        _ = layer(
            sample_hidden,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
        )[0]

    if prev_mode:
        layer.train()

    for h in hooks:
        h.remove()

    return captured


def _compare_linear_outputs(orig_linear, quant_linear, sample_input, name, device):
    """Compute output-diff metrics between original and quantized linear modules."""
    if sample_input is None or sample_input.numel() == 0:
        return {
            'name': name,
            'status': 'no_input',
        }

    compare_device = torch.device(device) if torch.cuda.is_available() and str(device).startswith('cuda') else torch.device('cpu')

    orig_linear = copy.deepcopy(orig_linear).to(compare_device).eval()
    quant_linear = copy.deepcopy(quant_linear).to(compare_device).eval()

    with torch.no_grad():
        x = sample_input.to(compare_device)
        if hasattr(orig_linear, 'weight') and isinstance(orig_linear.weight, torch.Tensor):
            x = x.to(orig_linear.weight.dtype)
        y_orig = orig_linear(x)
        y_quant = quant_linear(x)

    y_orig_f = y_orig.float().reshape(-1)
    y_quant_f = y_quant.float().reshape(-1)
    diff = y_orig_f - y_quant_f

    mse = float((diff.pow(2).mean()).item())
    rmse = float(torch.sqrt(diff.pow(2).mean()).item())
    mean_abs_diff = float(diff.abs().mean().item())
    max_abs_diff = float(diff.abs().max().item())
    ref_rms = float(torch.sqrt((y_orig_f.pow(2).mean()) + 1e-12).item())
    relative_rmse = float(rmse / (ref_rms + 1e-12))

    denom = (y_orig_f.norm() * y_quant_f.norm()).clamp(min=1e-12)
    cosine = float(torch.dot(y_orig_f, y_quant_f).div(denom).item())

    return {
        'name': name,
        'status': 'ok',
        'num_tokens': int(sample_input.shape[0]),
        'in_features': int(sample_input.shape[-1]),
        'out_features': int(y_orig.shape[-1]) if y_orig.dim() > 1 else int(y_orig.numel()),
        'mse': mse,
        'rmse': rmse,
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
        'ref_rms': ref_rms,
        'relative_rmse': relative_rmse,
        'cosine_similarity': cosine,
    }


def quantize_deepseek_layer(
    layer, 
    idx, 
    cb, 
    args, 
    device, 
    pre_orig_emb, 
    orig_emb, 
    model_config
):
    """量化 DeepSeek 层，返回量化后的层"""
    # if idx == 2:
    #     print("layer is", layer)
    if check_exist(idx, layer, args):
        glog.info(f"Layer {idx} already fully quantized. Skipping.")
        # 即使已量化，也需要加载量化层替换原层
        # TODO: 实现加载逻辑
        # return layer
    
    # layer.to(device) 会创建新对象，需要保存并返回
    from model.deepseek_moe import DeepseekDecoderLayer
    mixed_layer = DeepseekDecoderLayer(model_config, idx).cpu()
    # if idx == 2:
    #     print("mixed_layer is", mixed_layer)
    #     exit()

    num_experts = model_config.n_routed_experts if hasattr(model_config, 'n_routed_experts') else model_config.n_experts

    num_shared_experts = model_config.n_shared_experts if hasattr(model_config, 'n_shared_experts') else 0

    with torch.no_grad():
        mixed_layer.self_attn.q_proj = copy.deepcopy(layer.self_attn.q_proj)
        mixed_layer.self_attn.k_proj = copy.deepcopy(layer.self_attn.k_proj)
        mixed_layer.self_attn.v_proj = copy.deepcopy(layer.self_attn.v_proj)
        mixed_layer.self_attn.o_proj = copy.deepcopy(layer.self_attn.o_proj)

        # 处理 gate（仅在 MoE 层存在）
        if hasattr(layer.mlp, 'gate') and layer.mlp.gate is not None:
            if args.quantize_gate:
                mixed_layer.mlp.gate.weight.data.copy_(
                    layer.mlp.gate.weight.data.to(mixed_layer.mlp.gate.weight.device)
                )
            else:
                gate_state = layer.mlp.gate.state_dict()
                torch.save(gate_state, f'{args.save_path}/{idx}_gate.pt')
                # mixed_layer.mlp.gate.weight.data.copy_(
                    # layer.mlp.gate.weight.data.to(mixed_layer.mlp.gate.weight.device)
                # )
                mixed_layer.mlp.gate = copy.deepcopy(layer.mlp.gate)
                glog.info(f'Saved gate layer {idx} in original precision (not quantized)')
                for param in mixed_layer.mlp.gate.parameters():
                    param.requires_grad = False
        
        # routed experts (仅在 MoE 层存在)
        if hasattr(layer.mlp, 'experts') and layer.mlp.experts is not None:
            for i in range(num_experts):
                expert = layer.mlp.experts[i]
                mixed_expert = mixed_layer.mlp.experts[i]
                # 复制权重数据而不是替换模块
                mixed_expert.gate_proj.weight.data.copy_(expert.gate_proj.weight.data)
                mixed_expert.up_proj.weight.data.copy_(expert.up_proj.weight.data)
                mixed_expert.down_proj.weight.data.copy_(expert.down_proj.weight.data)
        
        if hasattr(layer.mlp, 'shared_experts') and layer.mlp.shared_experts is not None:
            # 复制权重数据而不是替换模块
            mixed_layer.mlp.shared_experts.gate_proj.weight.data.copy_(layer.mlp.shared_experts.gate_proj.weight.data)
            mixed_layer.mlp.shared_experts.up_proj.weight.data.copy_(layer.mlp.shared_experts.up_proj.weight.data)
            mixed_layer.mlp.shared_experts.down_proj.weight.data.copy_(layer.mlp.shared_experts.down_proj.weight.data)
        
        # Dense MLP 层（前几层可能是 Dense 而不是 MoE）
        if (not hasattr(layer.mlp, 'experts') or layer.mlp.experts is None) and hasattr(layer.mlp, 'gate_proj'):
            glog.info(f"Layer {idx} is Dense MLP (not MoE)")
            mixed_layer.mlp.gate_proj.weight.data.copy_(layer.mlp.gate_proj.weight.data)
            mixed_layer.mlp.up_proj.weight.data.copy_(layer.mlp.up_proj.weight.data)
            mixed_layer.mlp.down_proj.weight.data.copy_(layer.mlp.down_proj.weight.data)
        
        if hasattr(layer, 'input_layernorm') and hasattr(mixed_layer, 'input_layernorm'):
            mixed_layer.input_layernorm.weight.data.copy_(
                layer.input_layernorm.weight.data.to(mixed_layer.input_layernorm.weight.device)
            )

        if hasattr(layer, 'post_attention_layernorm') and hasattr(mixed_layer, 'post_attention_layernorm'):
            mixed_layer.post_attention_layernorm.weight.data.copy_(
                layer.post_attention_layernorm.weight.data.to(mixed_layer.post_attention_layernorm.weight.device)
            )

    quant_order = finetune_deepseek.get_deepseek_quant_order(mixed_layer)
    # print(quant_order)

    # 保存原始线性层副本用于输出误差对比
    orig_linears = {}
    for linear_attr, name in quant_order:
        try:
            orig_linears[name] = copy.deepcopy(attrgetter(linear_attr)(mixed_layer)).cpu()
        except Exception as e:
            glog.warning(f'Failed to snapshot original linear for {name} ({linear_attr}): {e}')

    linear_input_samples = {}
    if args.compare_linear_outputs:
        try:
            layer_for_hook = mixed_layer.to(device)
            linear_input_samples = _collect_linear_inputs_for_compare(
                layer_for_hook,
                quant_order,
                pre_orig_emb,
                args,
                device,
            )
            mixed_layer = layer_for_hook.cpu()
            glog.info(
                f'Layer {idx} collected activation samples for {len(linear_input_samples)}/{len(quant_order)} target linears'
            )
        except Exception as e:
            glog.warning(f'Layer {idx} failed to collect linear input samples: {e}')
            mixed_layer = mixed_layer.cpu()
    
    glog.info(f'Starting quantization for DeepSeek layer {idx} with {len(quant_order)} target modules.')
    
    # 调用 finetune_deepseek 中的核心量化与微调函数
    # 使用 quant_order 量化所有层（包括 MLP/MoE）
    glog.info(f"Before quantization: q_proj type = {type(mixed_layer.self_attn.q_proj).__name__}")
    glog.info(f"Quantization order for layer {idx}: {[name for _, name in quant_order]}")
    finetune_deepseek.quantize_finetune_moe_decoder_layer(
        mixed_layer,
        quant_order,
        idx,
        cb,
        args,
        device,
        pre_orig_emb,
        orig_emb
    )
    glog.info(f"After quantization: q_proj type = {type(mixed_layer.self_attn.q_proj).__name__}")

    linear_output_report = []
    if args.compare_linear_outputs:
        for linear_attr, name in quant_order:
            sample_input = linear_input_samples.get(name, None)
            try:
                quant_linear = attrgetter(linear_attr)(mixed_layer)
                if name not in orig_linears:
                    linear_output_report.append({'name': name, 'status': 'missing_original_snapshot'})
                    continue
                metrics = _compare_linear_outputs(
                    orig_linears[name],
                    quant_linear,
                    sample_input,
                    name,
                    device,
                )
                linear_output_report.append(metrics)
                if metrics.get('status') == 'ok':
                    glog.info(
                        f"Layer {idx} {name} output diff: max_abs={metrics['max_abs_diff']:.6e}, "
                        f"mean_abs={metrics['mean_abs_diff']:.6e}, rel_rmse={metrics['relative_rmse']:.6e}, "
                        f"cos={metrics['cosine_similarity']:.6f}"
                    )
                    if metrics['relative_rmse'] > args.compare_rel_rmse_warn or metrics['cosine_similarity'] < args.compare_cosine_warn:
                        glog.warning(
                            f"Layer {idx} {name} has large quantization error: "
                            f"rel_rmse={metrics['relative_rmse']:.6e} (warn>{args.compare_rel_rmse_warn}), "
                            f"cos={metrics['cosine_similarity']:.6f} (warn<{args.compare_cosine_warn})"
                        )
                else:
                    glog.warning(f"Layer {idx} {name} output diff skipped: {metrics.get('status')}")
            except Exception as e:
                glog.warning(f'Layer {idx} failed output compare for {name}: {e}')
                linear_output_report.append({'name': name, 'status': f'compare_failed: {str(e)}'})
    
    # 保存层归一化参数
    layernorm_dict = {}
    if hasattr(mixed_layer, 'input_layernorm'):
        layernorm_dict['input_layernorm'] = mixed_layer.input_layernorm.weight
    if hasattr(mixed_layer, 'post_attention_layernorm'):
        layernorm_dict['post_attention_layernorm'] = mixed_layer.post_attention_layernorm.weight
        
    if layernorm_dict:
        torch.save(layernorm_dict, f'{args.save_path}/{idx}_layernorm.pt')
    
    # 将量化后的层移回 CPU 并返回
    quantized_layer = mixed_layer.cpu()
    
    # 立即清理 GPU 显存
    del mixed_layer
    utils.clean()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return quantized_layer, linear_output_report  # 返回量化后的层及误差报告


def detect_model_type(model_config):
    """自动检测模型类型"""
    config_dict = model_config.to_dict()
    
    if 'architectures' in config_dict:
        arch = config_dict['architectures'][0].lower()
        if 'deepseek' in arch:
            return 'deepseek'
        elif 'mistral' in arch:
            return 'mistral'
        elif 'llama' in arch:
            return 'llama'
        elif 'mixtral' in arch:
            return 'mixtral'
            
    if 'num_local_experts' in config_dict:
        return 'mixtral'
    
    # 默认使用 deepseek
    return 'deepseek'


def pick_best_cuda_device():
    """Pick CUDA device with maximum free memory."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None

    best_idx = 0
    best_free = -1
    for i in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        glog.info(
            f"CUDA:{i} free={free_bytes / (1024**3):.2f}GB / total={total_bytes / (1024**3):.2f}GB"
        )
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = i
    return torch.device(f'cuda:{best_idx}')


def main(args):
    torch.set_num_threads(args.num_cpu_threads)
    # dtype_ = torch.float64 if args.use_fp64 else torch.float32
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载码本
    cb = codebook.get_codebook(args.codebook)
    
    # 加载模型和tokenizer
    glog.info(f'Loading model from {args.base_model}')
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=None  # 统一先加载到CPU，避免CPU/GPU混放导致设备不一致
    )
    custom_config = DeepseekConfig(**hf_model.config.to_dict())
    # Quantization starts from dense FP weights, so force FP architecture here.
    if hasattr(custom_config, 'quip_params'):
        custom_config.quip_params = None
    model = DeepseekForCausalLM(custom_config)
    # print("hf_model", hf_model) 
    # print("model", model)
    load_info = model.load_state_dict(hf_model.state_dict())
    if len(load_info.missing_keys) > 0:
        glog.warning(f"Missing keys when loading HF weights into custom model: {len(load_info.missing_keys)}")
    
    if len(load_info.unexpected_keys) > 0:
        glog.warning(f"Unexpected keys when loading HF weights into custom model: {len(load_info.unexpected_keys)}")
    del hf_model
    # for i, layer in enumerate(model.model.layers):
    #     if hasattr(layer.mlp, "gate"):
    #         print(f"Layer {i} gate weight dtype: {layer.mlp.gate.weight.dtype}")

    # input_texts = "The capital of France is"
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True) 
    # output_text =  tokenizer.decode(model.generate(**tokenizer(input_texts, return_tensors='pt'), max_new_tokens=5)[0], skip_special_tokens=True)
    # glog.info(f'Model loaded successfully. Example output: {output_text}')
    model_dtype = next(model.parameters()).dtype
    glog.info(f'Model dtype: {model_dtype}')
    
    # 检测模型类型
    if args.model_type == 'auto':
        args.model_type = detect_model_type(model.config)
    glog.info(f'Detected model type: {args.model_type}')
    
    # 保存配置
    all_config = {
        'quant_args': vars(args),
        'model_config': model.config.to_dict(),
        'model_type': args.model_type
    }
    
    quip_params = {
        'lora_rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'codebook': args.codebook,
        'codebook_version': cb.version if hasattr(cb, 'version') else 'unknown',
        'codesz': cb.codesz if hasattr(cb, 'codesz') else 1,
        'idx_dtype': str(cb.idx_dtype) if hasattr(cb, 'idx_dtype') else 'int32',
        'packsz': cb.packsz if hasattr(cb, 'packsz') else 1,
        'resid_scale_override': args.resid_scale_override,
    }
    
    if 'quip_params' not in all_config['model_config'] or all_config['model_config']['quip_params'] is None:
        all_config['model_config']['quip_params'] = {}
    all_config['model_config']['quip_params'].update(quip_params)
    
    config_path = os.path.join(args.save_path, 'config.pt')
    torch.save(all_config, config_path)
    
    config_json_path = os.path.join(args.save_path, 'config.json')
    with open(config_json_path, 'w') as f:
        json.dump(all_config, f, indent=2, default=str)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # texts = "The capital of France is"
    # inputs = tokenizer(texts, return_tensors='pt')
    # test_device = next(model.parameters()).device
    # inputs = {k: v.to(test_device) for k, v in inputs.items()}
    
    # # 使用 generate() 方法生成文本
    # glog.info('Testing model output before quantization...')
    # with torch.no_grad():
    #     generated_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=20,
    #         do_sample=False,
    #         temperature=1.0,
    #         top_p=1.0,
    #     )
    # out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # glog.info(f'Example output before quantization: {out_text}')
    
    glog.info('Model and tokenizer loaded successfully')
    
    # 创建校准数据集
    glog.info('Creating calibration dataset...')
    devset = utils.sample_rp1t(
        tokenizer, 
        args.devset_size, 
        args.ctx_size,
        args.sample_proc
    )
    glog.info(f'Dataset created with {len(devset)} samples')
    
    # 确定设备
    if torch.cuda.is_available():
        device_type = 'cuda'
        device_id = 'cuda:0'
        glog.info('Using GPU: cuda:0 for debugging')
    else:
        device_type = 'cpu'
        device_id = 'cpu'
        glog.warning('No GPU found, using CPU only')
    
    # 准备嵌入缓存
    orig_emb = model.model.embed_tokens(devset).to(model_dtype)
    
    # 单线程下，我们只需要当前层的输入和输出缓存
    current_emb = orig_emb.clone().cpu()
    next_emb = torch.zeros_like(current_emb).cpu()
    
    # 准备注意力掩码和位置ID
    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        current_emb[:args.batch_size], 0)
    
    num_layers = len(model.model.layers)
    glog.info(f'Starting single-threaded quantization of {num_layers} layers')
    per_layer_linear_diff_reports = []
    
    for i in tqdm(range(num_layers)):
        glog.info(f'Processing layer {i+1}/{num_layers}')
        
        # 清理内存
        utils.clean()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 计算原始层的输出（无论是否微调都必须做，用于下一层输入传播与诊断）
        if True:
            st = time.time()
            next_emb.zero_()
            
            # 移动相关张量到设备
            if device_type == 'cuda':
                position_ids_dev = position_ids.to(device_id)
                attention_mask_dev = attention_mask.to(device_id)
                layer_dev = model.model.layers[i].to(device_id)
            else:
                position_ids_dev = position_ids
                attention_mask_dev = attention_mask
                layer_dev = model.model.layers[i]
            
            # 分批计算原始输出
            num_batches = args.devset_size // args.batch_size
            for j in range(num_batches):
                batch_start = args.batch_size * j
                batch_end = args.batch_size * (j + 1)
                
                # 计算层输出
                batch_emb = current_emb[batch_start:batch_end]
                if device_type == 'cuda':
                    batch_emb = batch_emb.to(device_id).to(model_dtype)
                
                with torch.no_grad():
                    layer_output = layer_dev(
                        batch_emb,
                        position_ids=position_ids_dev,
                        attention_mask=attention_mask_dev,
                        use_cache=False,
                        output_attentions=False
                    )[0]
                
                # 保存到下一层的缓存
                next_emb[batch_start:batch_end] = layer_output.cpu()
                
                # 清理中间变量
                del batch_emb, layer_output
            
            # 计算均方值统计
            orig_msv = current_emb.float().norm()**2 / current_emb.numel()
            target_msv = next_emb.float().norm()**2 / next_emb.numel()
            
            # 将层移回CPU
            model.model.layers[i] = layer_dev.cpu()
            del layer_dev
            
            if device_type == 'cuda':
                position_ids_dev = position_ids_dev.cpu()
                attention_mask_dev = attention_mask_dev.cpu()
            
            utils.clean()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            glog.info(
                f'Computed original embedding for layer {i} in {time.time()-st:.2f}s, '
                f'pre msv {orig_msv:.6f}, post msv {target_msv:.6f}'
            )
        
        # 根据模型类型选择量化函数
        if args.model_type == 'deepseek':
            quantize_func = quantize_deepseek_layer
        else:
            glog.error(f'Unsupported model type: {args.model_type}. This script is optimized for DeepSeek.')
            return
        
        original_layer = model.model.layers[i]  # 保存引用
        quantized_layer, linear_diff_report = quantize_func(
            original_layer,
            i,
            cb,
            args,
            device_id,
            current_emb,
            next_emb,
            model.config,
        )
        per_layer_linear_diff_reports.append({
            'layer_idx': i,
            'linear_output_diffs': linear_diff_report,
        })
        
        # 替换层并立即删除原始层引用
        model.model.layers[i] = quantized_layer
        del original_layer  # 显式删除原始层引用，帮助垃圾回收
        
        # 验证层已被替换
        glog.info(f"Layer {i} quantized. Checking first linear layer type...")

        if hasattr(model.model.layers[i].self_attn, 'q_proj'):
            layer_type = type(model.model.layers[i].self_attn.q_proj).__name__
            glog.info(f"  Layer {i} q_proj type: {layer_type}")
        
        # 强制清理 GPU 显存和运行垃圾回收
        import gc
        gc.collect()  # 强制Python垃圾回收
        utils.clean()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 等待所有CUDA操作完成
        
        # 将 next_emb 复制给 current_emb，为下一层做准备
        current_emb.copy_(next_emb)
    
    # 保存嵌入层和输出层
    glog.info('Saving embedding and output layers...')
    
    # 嵌入层
    embed_state = {
        'weight': model.model.embed_tokens.weight,
    }
    torch.save(embed_state, os.path.join(args.save_path, 'embed.pt'))
    
    # 输出层（语言模型头）
    lm_head_state = {
        'weight': model.lm_head.weight,
    }
    torch.save(lm_head_state, os.path.join(args.save_path, 'lm_head.pt'))
    
    # 最终层归一化（如果有）
    if hasattr(model.model, 'norm'):
        norm_state = {
            'weight': model.model.norm.weight,
        }
        torch.save(norm_state, os.path.join(args.save_path, 'final_norm.pt'))
    
    glog.info('Quantization completed successfully!')
    glog.info(f'All quantized weights saved to {args.save_path}')

    if args.compare_linear_outputs:
        diff_report_path = os.path.join(args.save_path, 'linear_output_diff_report.json')
        with open(diff_report_path, 'w') as f:
            json.dump(per_layer_linear_diff_reports, f, indent=2)
        glog.info(f'Linear output diff report saved to {diff_report_path}')

        suspicious = []
        for layer_item in per_layer_linear_diff_reports:
            layer_idx = layer_item['layer_idx']
            for item in layer_item['linear_output_diffs']:
                if item.get('status') != 'ok':
                    continue
                suspicious.append((
                    float(item.get('relative_rmse', 0.0)),
                    float(item.get('max_abs_diff', 0.0)),
                    layer_idx,
                    item.get('name', 'unknown')
                ))
        suspicious.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for rel_rmse, max_abs, layer_idx, name in suspicious[:10]:
            glog.info(
                f'Top output diff layer {layer_idx} {name}: rel_rmse={rel_rmse:.6e}, max_abs_diff={max_abs:.6e}'
            )
    
    # 清理中间变量但保留量化后的模型
    glog.info('Cleaning up intermediate variables before generation test...')
    del current_emb, next_emb, orig_emb
    if 'devset' in locals():
        del devset
    if 'position_ids' in locals():
        del position_ids
    if 'attention_mask' in locals():
        del attention_mask
    if 'position_ids_dev' in locals():
        del position_ids_dev
    if 'attention_mask_dev' in locals():
        del attention_mask_dev
    
    # 强制垃圾回收和GPU缓存清理
    import gc
    gc.collect()
    gc.collect()  # 运行两次确保彻底清理
    utils.clean()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(model) 
    # 测试生成（使用已量化的模型）
    if args.skip_generation_test or not args.run_generation_test:
        glog.info('Skipping generation test (default behavior to save GPU memory)')
        glog.info('Use --run_generation_test flag if you want to test generation')
    else:
        glog.info('Testing generation with quantized model...')
        glog.info(f'Available GPUs: {torch.cuda.device_count()}')

        # QUIP 自定义解压算子仅支持 CUDA，生成前必须将模型与输入迁移到 GPU。
        if not torch.cuda.is_available():
            raise RuntimeError(
                'Generation test requires CUDA because quip_lib::decompress_packed_e8p is CUDA-only.'
            )

        gen_device = pick_best_cuda_device()
        if gen_device is None:
            glog.warning('No CUDA device found. Skipping generation test.')
        else:
            try:
                glog.info(f'Moving quantized model to {gen_device} for generation test...')
                torch.cuda.empty_cache()
                model = model.to(gen_device)
                model.eval()

                texts = "The capital of France is"
                inputs = tokenizer(texts, return_tensors='pt')
                inputs = {k: v.to(gen_device) for k, v in inputs.items()}

                glog.info('Testing model output after quantization...')
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                    )
                out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                glog.info(f'Example output after quantization: {out_text}')
            except torch.OutOfMemoryError as e:
                glog.warning(f'Skipping generation test due to CUDA OOM on {gen_device}: {e}')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 记录命令行参数
    args_file = os.path.join(args.save_path, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 运行主函数
    main(args)