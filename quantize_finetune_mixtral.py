import argparse
import os
import time
import json
from tqdm import tqdm
import copy

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune_mixtral, quip
from lib.linear import FusedLinear

from model.mixtral_moe import MixtralDecoderLayer
MIXTRAL_AVAILABLE = True

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
                    choices=['auto', 'llama', 'mixtral', 'mistral'],
                    help='Model type to quantize')
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--incoh_mode',
                    default='had',
                    type=str,
                    choices=['had', 'kron'])
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


# class FusedMoELinear(nn.Module):
#     """专门为MoE设计的融合线性层"""
#     def __init__(self, num_experts, in_features, out_features_per_expert, total_out_features, bias=False):
#         super().__init__()
#         self.num_experts = num_experts
#         self.in_features = in_features
#         self.out_features_per_expert = out_features_per_expert
#         self.total_out_features = total_out_features
        
#         # 将所有专家的权重存储在单个张量中
#         self.weight = nn.Parameter(torch.empty(total_out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(total_out_features))
#         else:
#             self.register_parameter('bias', None)
    
#     def get_expert_weight(self, expert_idx):
#         """获取特定专家的权重切片"""
#         start = expert_idx * self.out_features_per_expert
#         end = start + self.out_features_per_expert
#         return self.weight[start:end], self.bias[start:end] if self.bias is not None else None


def check_exist(idx, args):
    """检查层的量化文件是否已存在"""
    if args.model_type in ['mixtral']:
        for suffix in ['q', 'k', 'v', 'o']:
            if not os.path.exists(f'{args.save_path}/{idx}_{suffix}.pt'):
                return False
        
        # 检查专家层量化文件
        for i in range(8):
            for w_name in ['w1', 'w2', 'w3']:
                if not os.path.exists(f'{args.save_path}/{idx}_expert{i}_{w_name}.pt'):
                    return False
        
        # 检查 gate 和 layernorm
        if not os.path.exists(f'{args.save_path}/{idx}_gate.pt'):
            return False
        if not os.path.exists(f'{args.save_path}/{idx}_layernorm.pt'):
            return False
        
        return True
    else:
        suffixes = ['qkv', 'o', 'up', 'down', 'layernorm']
        for suffix in suffixes:
            test = f'{args.save_path}/{idx}_{suffix}.pt'
            if not os.path.exists(test):
                return False
        return True



def quantize_mixtral_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb,
                           model_config):
    """量化Mixtral层"""
    if check_exist(idx, args):
        return
    
    from model.mixtral_moe import MixtralDecoderLayer
    mixed_layer = MixtralDecoderLayer(model_config, idx).cpu()
    
    num_experts = model_config.num_local_experts
    
    with torch.no_grad():
        mixed_layer.self_attn.q_proj = layer.self_attn.q_proj
        mixed_layer.self_attn.k_proj = layer.self_attn.k_proj
        mixed_layer.self_attn.v_proj = layer.self_attn.v_proj
        mixed_layer.self_attn.o_proj = layer.self_attn.o_proj
        
        if args.quantize_gate:
            mixed_layer.block_sparse_moe.gate = layer.block_sparse_moe.gate
        else:
            gate_state = layer.block_sparse_moe.gate.state_dict()
            torch.save(gate_state, f'{args.save_path}/{idx}_gate.pt')
            mixed_layer.block_sparse_moe.gate = copy.deepcopy(layer.block_sparse_moe.gate)
            for param in mixed_layer.block_sparse_moe.gate.parameters():
                param.requires_grad = False
        
        for i in range(num_experts):
            expert = layer.block_sparse_moe.experts[i]
            mixed_expert = mixed_layer.block_sparse_moe.experts[i]
            
            mixed_expert.w1 = expert.w1
            mixed_expert.w2 = expert.w2
            mixed_expert.w3 = expert.w3
        
        # 层归一化
        mixed_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        mixed_layer.post_attention_layernorm.weight.copy_(
            layer.post_attention_layernorm.weight)
    
    # ============================================================================
    # 量化 Q/K/V/O 四个注意力投影层
    # ============================================================================
    glog.info(f'Quantizing self_attn Q/K/V/O projections for layer {idx}')
    
    # 量化 Q, K, V, O 四个投影
    linear_pairs = [
        ('self_attn.q_proj', 'q'),
        ('self_attn.k_proj', 'k'),
        ('self_attn.v_proj', 'v'),
        ('self_attn.o_proj', 'o'),
    ]
    
    if not args.quantize_gate:
        gate_state = layer.block_sparse_moe.gate.state_dict()
        torch.save(gate_state, f'{args.save_path}/{idx}_gate.pt')
        glog.info(f'Saved gate layer {idx} in FP16 (not quantized)')
    
    for i in range(num_experts):
        linear_pairs.extend([
            (f'block_sparse_moe.experts.{i}.w1', f'expert{i}_w1'),
            (f'block_sparse_moe.experts.{i}.w2', f'expert{i}_w2'),
            (f'block_sparse_moe.experts.{i}.w3', f'expert{i}_w3'),
        ])

    finetune_mixtral.quantize_finetune_moe_decoder_layer(
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
    
    del mixed_layer
    utils.clean()  # 添加显存清理



def detect_model_type(model_config):
    """自动检测模型类型"""
    config_dict = model_config.to_dict()
    
    if 'num_local_experts' in config_dict:
        return 'mixtral'
    elif 'architectures' in config_dict:
        arch = config_dict['architectures'][0].lower()
        if 'mistral' in arch:
            return 'mistral'
        elif 'llama' in arch:
            return 'llama'
    
    # 默认使用llama
    return 'llama'


def main(args):
    torch.set_num_threads(args.num_cpu_threads)
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载码本
    cb = codebook.get_codebook(args.codebook)
    
    # 加载模型和tokenizer
    glog.info(f'Loading model from {args.base_model}')
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        device_map='cpu'  # 先加载到CPU
    )
    
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
    
    if 'quip_params' not in all_config['model_config']:
        all_config['model_config']['quip_params'] = {}
    all_config['model_config']['quip_params'].update(quip_params)
    
    config_path = os.path.join(args.save_path, 'config.pt')
    torch.save(all_config, config_path)
    
    # 保存配置文件为json（便于阅读）
    config_json_path = os.path.join(args.save_path, 'config.json')
    with open(config_json_path, 'w') as f:
        json.dump(all_config, f, indent=2, default=str)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    glog.info('Model and tokenizer loaded successfully')
    
    # 创建校准数据集
    glog.info('Creating calibration dataset...')
    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)
    glog.info(f'Dataset created with {len(devset)} samples')
    
    # 获取GPU数量
    nproc = torch.cuda.device_count()
    if nproc == 0:
        glog.warning('No GPU found, using CPU only')
        nproc = 1
        device_type = 'cpu'
    else:
        device_type = 'cuda'
        glog.info(f'Found {nproc} GPUs')
    
    # 准备嵌入缓存
    orig_emb = model.model.embed_tokens(devset)
    orig_emb_cache = [orig_emb]
    
    # 为每个进程创建缓存
    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                       dtype=orig_emb_cache[0].dtype,
                       device='cpu'))  # 保持在CPU上以节省GPU内存
    
    # 准备注意力掩码和位置ID
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
    
    for i in tqdm(range(num_layers)):
    # For testing:
    # for i in tqdm([17, 18, 19]):
        glog.info(f'Processing layer {i+1}/{num_layers} on device {cur_device}')
        
        # 等待前一个进程完成
        if proc_list[cur_device] is not None:
            proc_list[cur_device].join()
            if cur_device == 0:
                # 更新第一个设备的嵌入缓存
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        
        # 如果下一设备有进程，也等待
        if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
            proc_list[cur_device + 1].join()
        
        # 清理内存
        utils.clean()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            glog.info(
                f'Computed original embedding for layer {i} in {time.time()-st:.2f}s, '
                f'pre msv {orig_msv:.6f}, post msv {target_msv:.6f}'
            )
        
        # 根据模型类型选择量化函数
        if args.model_type == 'mixtral':
            if not MIXTRAL_AVAILABLE:
                glog.error('Mixtral model module not available. Please install it or check model/mixtral_moe.py')
                return
            
            quantize_func = quantize_mixtral_layer
        else:
            glog.error('Llama model module not available. Please install it or check model/llama.py')
            return
            
            quantize_func = None
        
        # 启动量化进程
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
    glog.info('Waiting for all quantization processes to finish...')
    for p in proc_list:
        if p is not None:
            p.join()
    
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
    
    # 运行主函数
    main(args)