import argparse
import os
import sys
import time
import json
from tqdm import tqdm
import copy

import glog

# Add parent directory to path to import lib module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import quip
from lib.algo import finetune_qwen15 as finetune_qwen
from lib.linear import FusedLinear

from model.qwen15_moe import Qwen2MoeDecoderLayer as QwenMoeDecoderLayer
QWEN_AVAILABLE = True

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
parser.add_argument('--shared_hessian_path', type=str, help='Hessian path for shared expert')
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--model_type', default='auto', type=str,
                    choices=['auto', 'llama', 'mixtral', 'mistral', 'qwen'],
                    help='Model type to quantize')
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--had_block_size',
                    default=64,
                    type=int,
                    help='Block size for blockwise Hadamard transform (must be power of 2)')
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
parser.add_argument('--expert_batch_size', default=4, type=int,
                    help='Number of experts to process at once (for memory control)')
parser.add_argument('--quantize_gate', action='store_true',
                    help='Whether to quantize the gate network')
parser.add_argument('--quantize_shared_expert', action='store_true', default=True,
                    help='Whether to quantize the shared expert')
parser.add_argument('--gate_precision', default='bf16', type=str,
                    choices=['fp32', 'fp16', 'bf16'],
                    help='Precision for gate network if not quantized')


def check_exist_qwen(idx, args):
    """
    检查 Qwen1.5 MoE 层的量化文件是否已存在
    
    Qwen1.5 MoE 需要检查的文件：
    - Attention: q_proj, k_proj, v_proj, o_proj
    - Gate: gate (可选)
    - Routed Experts: expert{i}_gate_proj, expert{i}_up_proj, expert{i}_down_proj (60个)
    - Shared Expert: shared_expert_gate_proj, shared_expert_up_proj, shared_expert_down_proj (可选)
    - Shared Expert Gate: shared_expert_gate (可选)
    - LayerNorm: layernorm
    """
    # 检查注意力层
    for suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if not os.path.exists(f'{args.save_path}/{idx}_{suffix}.pt'):
            glog.info(f'Missing attention file: {idx}_{suffix}.pt')
            return False
    
    # 检查门控网络（如果需要量化）
    if args.quantize_gate:
        if not os.path.exists(f'{args.save_path}/{idx}_gate.pt'):
            glog.info(f'Missing gate file: {idx}_gate.pt')
            return False
    
    # 检查路由专家层（通常60个）
    num_routed_experts = 60  # Qwen1.5 MoE 默认
    for i in range(num_routed_experts):
        for w_name in ['gate_proj', 'up_proj', 'down_proj']:
            expert_file = f'{args.save_path}/{idx}_expert{i}_{w_name}.pt'
            if not os.path.exists(expert_file):
                glog.info(f'Missing expert file: {expert_file}')
                return False
    
    # 检查共享专家（如果需要量化）
    if args.quantize_shared_expert:
        for w_name in ['gate_proj', 'up_proj', 'down_proj']:
            shared_file = f'{args.save_path}/{idx}_shared_expert_{w_name}.pt'
            if not os.path.exists(shared_file):
                glog.info(f'Missing shared expert file: {shared_file}')
                return False
        
        # 检查共享专家门控
        if not os.path.exists(f'{args.save_path}/{idx}_shared_expert_gate.pt'):
            glog.info(f'Missing shared expert gate file: {idx}_shared_expert_gate.pt')
            return False
    
    # 检查层归一化
    if not os.path.exists(f'{args.save_path}/{idx}_layernorm.pt'):
        glog.info(f'Missing layernorm file: {idx}_layernorm.pt')
        return False
    
    glog.info(f'All files exist for layer {idx}')
    return True


def quantize_qwen_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb,
                        model_config):
    """
    量化 Qwen1.5 MoE 层
    
    Args:
        layer: 原始的 Qwen decoder layer
        idx: 层索引
        cb: codebook 对象
        args: 参数配置
        device: 计算设备
        pre_orig_emb: 输入嵌入
        orig_emb: 目标输出嵌入
        model_config: 模型配置
    """
    # 检查是否已经量化
    if check_exist_qwen(idx, args):
        glog.info(f'Layer {idx} already quantized, skipping...')
        return
    
    # 创建混合精度层
    mixed_layer = QwenMoeDecoderLayer(model_config, idx).cpu()
    
    # 获取专家数量
    num_routed_experts = getattr(model_config, 'n_routed_experts', 60)
    num_shared_experts = getattr(model_config, 'num_shared_experts', 1)
    
    glog.info(f'Quantizing Qwen layer {idx}:')
    glog.info(f'  - Routed experts: {num_routed_experts}')
    glog.info(f'  - Shared experts: {num_shared_experts}')
    glog.info(f'  - Quantize gate: {args.quantize_gate}')
    glog.info(f'  - Quantize shared expert: {args.quantize_shared_expert}')
    
    with torch.no_grad():
        # 复制注意力层（注意 Qwen 的 q/k/v 有 bias）
        mixed_layer.self_attn.q_proj = copy.deepcopy(layer.self_attn.q_proj)
        mixed_layer.self_attn.k_proj = copy.deepcopy(layer.self_attn.k_proj)
        mixed_layer.self_attn.v_proj = copy.deepcopy(layer.self_attn.v_proj)
        mixed_layer.self_attn.o_proj = copy.deepcopy(layer.self_attn.o_proj)
        
        # 处理门控网络
        if args.quantize_gate:
            glog.info(f'Will quantize gate network for layer {idx}')
            mixed_layer.mlp.gate = copy.deepcopy(layer.mlp.gate)
        else:
            # 保存门控网络为 FP16/BF16
            gate_state = layer.mlp.gate.state_dict()
            
            # 转换精度
            if args.gate_precision == 'fp16':
                gate_state = {k: v.half() for k, v in gate_state.items()}
            elif args.gate_precision == 'bf16':
                gate_state = {k: v.bfloat16() for k, v in gate_state.items()}
            
            torch.save(gate_state, f'{args.save_path}/{idx}_gate.pt')
            glog.info(f'Saved gate network for layer {idx} in {args.gate_precision} (not quantized)')
            
            mixed_layer.mlp.gate = copy.deepcopy(layer.mlp.gate)
            for param in mixed_layer.mlp.gate.parameters():
                param.requires_grad = False
        
        # 复制路由专家
        for i in range(num_routed_experts):
            expert = layer.mlp.experts[i]
            mixed_expert = mixed_layer.mlp.experts[i]
            
            mixed_expert.gate_proj = copy.deepcopy(expert.gate_proj)
            mixed_expert.up_proj = copy.deepcopy(expert.up_proj)
            mixed_expert.down_proj = copy.deepcopy(expert.down_proj)
        
        # 处理共享专家
        if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
            if args.quantize_shared_expert:
                glog.info(f'Will quantize shared expert for layer {idx}')
                mixed_layer.mlp.shared_expert = copy.deepcopy(layer.mlp.shared_expert)
                
                # 复制共享专家门控
                if hasattr(layer.mlp, 'shared_expert_gate'):
                    mixed_layer.mlp.shared_expert_gate = copy.deepcopy(layer.mlp.shared_expert_gate)
            else:
                # 保存共享专家为 FP16
                shared_expert_state = {
                    'gate_proj': layer.mlp.shared_expert.gate_proj.state_dict(),
                    'up_proj': layer.mlp.shared_expert.up_proj.state_dict(),
                    'down_proj': layer.mlp.shared_expert.down_proj.state_dict(),
                }
                torch.save(shared_expert_state, f'{args.save_path}/{idx}_shared_expert.pt')
                
                if hasattr(layer.mlp, 'shared_expert_gate'):
                    shared_gate_state = layer.mlp.shared_expert_gate.state_dict()
                    torch.save(shared_gate_state, f'{args.save_path}/{idx}_shared_expert_gate.pt')
                
                glog.info(f'Saved shared expert for layer {idx} in FP16 (not quantized)')
                
                mixed_layer.mlp.shared_expert = copy.deepcopy(layer.mlp.shared_expert)
                for param in mixed_layer.mlp.shared_expert.parameters():
                    param.requires_grad = False
                
                if hasattr(layer.mlp, 'shared_expert_gate'):
                    mixed_layer.mlp.shared_expert_gate = copy.deepcopy(layer.mlp.shared_expert_gate)
                    for param in mixed_layer.mlp.shared_expert_gate.parameters():
                        param.requires_grad = False
        
        # 复制层归一化
        mixed_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        mixed_layer.post_attention_layernorm.weight.copy_(
            layer.post_attention_layernorm.weight)
    
    # 构建量化顺序
    glog.info(f'Building quantization order for layer {idx}')
    linear_pairs = [
        ('self_attn.q_proj', 'q_proj'),
        ('self_attn.k_proj', 'k_proj'),
        ('self_attn.v_proj', 'v_proj'),
        ('self_attn.o_proj', 'o_proj'),
    ]
    
    # 添加门控网络（如果需要量化）
    if args.quantize_gate:
        linear_pairs.append(('mlp.gate', 'gate'))
    
    # 添加路由专家
    for i in range(num_routed_experts):
        linear_pairs.extend([
            (f'mlp.experts.{i}.gate_proj', f'expert{i}_gate_proj'),
            (f'mlp.experts.{i}.up_proj', f'expert{i}_up_proj'),
            (f'mlp.experts.{i}.down_proj', f'expert{i}_down_proj'),
        ])
    
    # 添加共享专家（如果需要量化）
    if args.quantize_shared_expert and hasattr(layer.mlp, 'shared_expert'):
        linear_pairs.extend([
            ('mlp.shared_expert.gate_proj', 'shared_expert_gate_proj'),
            ('mlp.shared_expert.up_proj', 'shared_expert_up_proj'),
            ('mlp.shared_expert.down_proj', 'shared_expert_down_proj'),
        ])
        
        # 添加共享专家门控
        if hasattr(layer.mlp, 'shared_expert_gate'):
            linear_pairs.append(('mlp.shared_expert_gate', 'shared_expert_gate'))
    
    glog.info(f'Total layers to quantize: {len(linear_pairs)}')
    
    # 执行量化和微调
    finetune_qwen.quantize_finetune_qwen_decoder_layer(
        mixed_layer, 
        linear_pairs, 
        idx, 
        cb, 
        args, 
        device,
        pre_orig_emb, 
        orig_emb
    )
    
    # 保存层归一化参数
    torch.save(
        {
            'input_layernorm': mixed_layer.input_layernorm.weight,
            'post_attention_layernorm': mixed_layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt')
    
    glog.info(f'Layer {idx} quantization completed')
    
    del mixed_layer
    utils.clean()


def detect_model_type(model_config):
    """自动检测模型类型"""
    config_dict = model_config.to_dict()
    
    # 检查是否是 Qwen MoE
    if 'n_routed_experts' in config_dict or 'num_experts_per_tok' in config_dict:
        model_name = config_dict.get('_name_or_path', '').lower()
        if 'qwen' in model_name:
            return 'qwen'
    
    # 检查是否是 Mixtral
    if 'num_local_experts' in config_dict:
        return 'mixtral'
    
    # 检查架构名称
    if 'architectures' in config_dict:
        arch = config_dict['architectures'][0].lower()
        if 'qwen' in arch:
            return 'qwen'
        elif 'mistral' in arch:
            return 'mistral'
        elif 'llama' in arch:
            return 'llama'
    
    # 默认使用 llama
    return 'llama'


def main(args):
    torch.set_num_threads(args.num_cpu_threads)
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载码本
    glog.info(f'Loading codebook: {args.codebook}')
    cb = codebook.get_codebook(args.codebook)
    
    # 加载模型和tokenizer
    glog.info(f'Loading model from {args.base_model}')
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        device_map='cpu',  # 先加载到CPU
        trust_remote_code=True  # Qwen 模型需要
    )
    
    # 检测模型类型
    if args.model_type == 'auto':
        args.model_type = detect_model_type(model.config)
    glog.info(f'Detected model type: {args.model_type}')
    
    # 验证是否是 Qwen MoE
    if args.model_type == 'qwen':
        if not hasattr(model.config, 'n_routed_experts'):
            glog.warning('Model does not appear to be Qwen1.5 MoE, continuing anyway...')
    
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
        'quantize_gate': args.quantize_gate,
        'quantize_shared_expert': args.quantize_shared_expert,
    }
    
    if 'quip_params' not in all_config['model_config']:
        all_config['model_config']['quip_params'] = {}
    all_config['model_config']['quip_params'].update(quip_params)
    
    config_path = os.path.join(args.save_path, 'config.pt')
    torch.save(all_config, config_path)
    
    config_json_path = os.path.join(args.save_path, 'config.json')
    with open(config_json_path, 'w') as f:
        json.dump(all_config, f, indent=2, default=str)
    
    glog.info('Configuration saved')
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    glog.info('Model and tokenizer loaded successfully')
    
    glog.info('Creating calibration dataset...')
    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)
    glog.info(f'Dataset created with {len(devset)} samples')
    
    nproc = torch.cuda.device_count()
    if nproc == 0:
        glog.warning('No GPU found, using CPU only')
        nproc = 1
        device_type = 'cpu'
    else:
        device_type = 'cuda'
        glog.info(f'Found {nproc} GPUs')
    
    glog.info('Computing initial embeddings...')
    orig_emb = model.model.embed_tokens(devset)
    orig_emb_cache = [orig_emb]
    
    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                       dtype=orig_emb_cache[0].dtype,
                       device='cpu'))  # 保持在CPU上以节省GPU内存
    
    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        orig_emb_cache[0][:args.batch_size], 0)
    
    # 分层处理
    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    
    num_layers = len(model.model.layers)
    glog.info(f'Starting quantization of {num_layers} layers')
    
    # 打印模型信息
    if args.model_type == 'qwen':
        num_routed_experts = getattr(model.config, 'n_routed_experts', 60)
        num_shared_experts = getattr(model.config, 'num_shared_experts', 1)
        glog.info(f'Qwen1.5 MoE Configuration:')
        glog.info(f'  - Routed experts per layer: {num_routed_experts}')
        glog.info(f'  - Shared experts per layer: {num_shared_experts}')
        glog.info(f'  - Total parameters to quantize per layer: {4 + num_routed_experts * 3 + num_shared_experts * 3}')
    
    for i in tqdm(range(num_layers)):
        glog.info(f'=' * 80)
        glog.info(f'Processing layer {i+1}/{num_layers} on device {cur_device}')
        
        # 等待前一个进程完成
        if proc_list[cur_device] is not None:
            glog.info(f'Waiting for previous process on device {cur_device} to finish...')
            proc_list[cur_device].join()
            if cur_device == 0:
                # 更新第一个设备的嵌入缓存
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        
        # 如果下一设备有进程，也等待
        if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
            glog.info(f'Waiting for next device process to finish...')
            proc_list[cur_device + 1].join()
        
        # 清理内存
        utils.clean()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 如果需要微调，计算原始层的输出
        if args.ft_epochs > 0:
            st = time.time()
            
            # 移动相关张量到设备
            if device_type == 'cuda':
                device_id = f'cuda:{cur_device}'
                position_ids_dev = position_ids.to(device_id)
                attention_mask_dev = attention_mask.to(device_id)
                layer_dev = model.model.layers[i].to(device_id)
            else:
                device_id = 'cpu'
                position_ids_dev = position_ids
                attention_mask_dev = attention_mask
                layer_dev = model.model.layers[i]
            
            # 分批计算原始输出
            num_batches = args.devset_size // args.batch_size
            glog.info(f'Computing original embeddings with {num_batches} batches...')
            
            for j in range(num_batches):
                batch_start = args.batch_size * j
                batch_end = args.batch_size * (j + 1)
                
                # 计算层输出
                batch_emb = orig_emb_cache[cur_device][batch_start:batch_end]
                if device_type == 'cuda':
                    batch_emb = batch_emb.to(device_id)
                
                with torch.no_grad():
                    layer_output = layer_dev(
                        batch_emb,
                        position_ids=position_ids_dev,
                        attention_mask=attention_mask_dev,
                        use_cache=False,
                        output_attentions=False
                    )[0]
                
                # 保存到缓存
                orig_emb_cache[cur_device + 1][batch_start:batch_end] = layer_output.cpu()
                
                # 清理中间变量
                del batch_emb, layer_output
                
                # 定期清理内存
                if j % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 计算均方值统计
            orig_msv = orig_emb_cache[cur_device].float().norm()**2 / orig_emb_cache[cur_device].numel()
            target_msv = orig_emb_cache[cur_device + 1].float().norm()**2 / orig_emb_cache[cur_device + 1].numel()
            
            # 将层移回CPU
            model.model.layers[i] = layer_dev.cpu()
            del layer_dev
            
            if device_type == 'cuda':
                position_ids_dev = position_ids_dev.cpu()
                attention_mask_dev = attention_mask_dev.cpu()
            
            utils.clean()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            glog.info(
                f'Computed original embedding for layer {i} in {time.time()-st:.2f}s, '
                f'pre msv {orig_msv:.6f}, post msv {target_msv:.6f}'
            )
        
        # 根据模型类型选择量化函数
        if args.model_type == 'qwen':
            if not QWEN_AVAILABLE:
                glog.error('Qwen model module not available. Please check model/qwen_moe.py')
                return
            
            quantize_func = quantize_qwen_layer
        elif args.model_type == 'mixtral':
            glog.error('Use the Mixtral quantization script for Mixtral models')
            return
        else:
            glog.error(f'Unsupported model type: {args.model_type}')
            return
        
        # 启动量化进程
        glog.info(f'Starting quantization process for layer {i}...')
        proc_list[cur_device] = mp.Process(
            target=quantize_func,
            args=(
                model.model.layers[i],
                i,
                cb,
                args,
                cur_device if device_type == 'cuda' else 'cpu',
                orig_emb_cache[cur_device],
                orig_emb_cache[cur_device + 1],
                model.config,
            )
        )
        proc_list[cur_device].start()
        
        # 更新设备索引
        cur_device = (cur_device + 1) % nproc
    
    # 等待所有进程完成
    glog.info('=' * 80)
    glog.info('Waiting for all quantization processes to finish...')
    for i, p in enumerate(proc_list):
        if p is not None:
            glog.info(f'Waiting for process {i+1}/{len(proc_list)}...')
            p.join()
    
    # 保存嵌入层和输出层
    glog.info('=' * 80)
    glog.info('Saving embedding and output layers...')
    
    # 嵌入层
    embed_state = {
        'weight': model.model.embed_tokens.weight,
    }
    torch.save(embed_state, os.path.join(args.save_path, 'embed.pt'))
    glog.info('Saved embedding layer')
    
    # 输出层（语言模型头）
    lm_head_state = {
        'weight': model.lm_head.weight,
    }
    torch.save(lm_head_state, os.path.join(args.save_path, 'lm_head.pt'))
    glog.info('Saved lm_head')
    
    # 最终层归一化
    if hasattr(model.model, 'norm'):
        norm_state = {
            'weight': model.model.norm.weight,
        }
        torch.save(norm_state, os.path.join(args.save_path, 'final_norm.pt'))
        glog.info('Saved final norm')
    
    glog.info('=' * 80)
    glog.info('Quantization completed successfully!')
    glog.info(f'All quantized weights saved to {args.save_path}')
    
    # 打印汇总信息
    glog.info('Quantization Summary:')
    glog.info(f'  - Model: {args.base_model}')
    glog.info(f'  - Model type: {args.model_type}')
    glog.info(f'  - Codebook: {args.codebook}')
    glog.info(f'  - Number of layers: {num_layers}')
    glog.info(f'  - Quantize gate: {args.quantize_gate}')
    glog.info(f'  - Quantize shared expert: {args.quantize_shared_expert}')
    glog.info(f'  - Fine-tuning epochs: {args.ft_epochs}')
    glog.info(f'  - Save path: {args.save_path}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')
    
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
    
    glog.info('Starting Qwen1.5 MoE quantization...')
    glog.info(f'Arguments: {json.dumps(vars(args), indent=2)}')
    
    # 运行主函数
    try:
        main(args)
    except Exception as e:
        glog.error(f'Quantization failed with error: {e}')
        import traceback
        traceback.print_exc()
        raise