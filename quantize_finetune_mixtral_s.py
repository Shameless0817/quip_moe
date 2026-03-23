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

# 动态导入模型模块
try:
    from model.mixtral_moe import MixtralDecoderLayer
    MIXTRAL_AVAILABLE = True
except ImportError:
    MIXTRAL_AVAILABLE = False

try:
    from model.llama import LlamaDecoderLayer
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

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
        # Mixtral有额外的专家文件
        suffixes = ['q','k','v','o','gate'] + [f'expert_{i}_w1' for i in range(8)] + \
                  [f'expert_{i}_w2' for i in range(8)] + ['layernorm'] + \
                  [f'expert_{i}_w3' for i in range(8)]
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
        weights = [
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight
        ]
        fused_qkv_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                     weights[0].shape[1],
                                     sum([_.shape[0] for _ in weights]),
                                     bias=False)
        cur = 0
        for w in weights:
            fused_qkv_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]
        mixed_layer.self_attn.qkv_proj = fused_qkv_proj
        mixed_layer.self_attn.o_proj = copy.deepcopy(layer.self_attn.o_proj)
        
        # 2. 门控网络处理
        if args.quantize_gate:
            mixed_layer.block_sparse_moe.gate = copy.deepcopy(layer.block_sparse_moe.gate)
        else:
            gate_state = layer.block_sparse_moe.gate.state_dict()
            torch.save(gate_state, f'{args.save_path}/{idx}_gate.pt')
            mixed_layer.block_sparse_moe.gate = copy.deepcopy(layer.block_sparse_moe.gate)
            for param in mixed_layer.block_sparse_moe.gate.parameters():
                param.requires_grad = False
        
        # 3. 处理所有专家
        for i in range(num_experts):
            expert = layer.block_sparse_moe.experts[i]
            mixed_expert = mixed_layer.block_sparse_moe.experts[i]
            
            # 直接复制 w1, w2, w3
            mixed_expert.w1 = copy.deepcopy(expert.w1)
            mixed_expert.w2 = copy.deepcopy(expert.w2)
            mixed_expert.w3 = copy.deepcopy(expert.w3)
        
        # 4. 层归一化
        mixed_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        mixed_layer.post_attention_layernorm.weight.copy_(
            layer.post_attention_layernorm.weight)
    
    # 构建量化层对列表
    linear_pairs = [
        ('self_attn.qkv_proj', 'q'),
        ('self_attn.o_proj', 'o'),
    ]
    
    # Gate层不量化（形状不兼容：8x128无法满足m%16==0的要求）
    # 保存原始gate权重
    if not args.quantize_gate:
        gate_state = layer.block_sparse_moe.gate.state_dict()
        torch.save(gate_state, f'{args.save_path}/{idx}_gate.pt')
        glog.info(f'Saved gate layer {idx} in FP16 (not quantized)')
    
    # 添加专家层（使用标准的 w1, w2, w3）
    # for i in range(num_experts):
    #     linear_pairs.extend([
    #         (f'block_sparse_moe.experts.{i}.w1', f'expert{i}_w1'),
    #         (f'block_sparse_moe.experts.{i}.w2', f'expert{i}_w2'),
    #         (f'block_sparse_moe.experts.{i}.w3', f'expert{i}_w3'),
    #     ])
    

    finetune_mixtral.quantize_finetune_moe_decoder_layer(
        mixed_layer, linear_pairs, idx, cb, args, device,
        pre_orig_emb, orig_emb
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



def main_debug(args):
    """
    更详细的调试版本，增加更多日志和断点
    """
    torch.set_num_threads(args.num_cpu_threads)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.hessian_path, exist_ok=True)
    
    glog.info('='*60)
    glog.info('DEBUG MODE - Single Threaded Quantization')
    glog.info('='*60)
    
    # 加载码本
    glog.info(f'Loading codebook: {args.codebook}')
    cb = codebook.get_codebook(args.codebook)
    glog.info(f'Codebook loaded: codesz={cb.codesz}, packsz={cb.packsz}')
    
    # 加载模型
    glog.info(f'Loading model from {args.base_model}')
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        device_map='cpu'
    )
    glog.info(f'Model loaded: {model.config.architectures}')
    glog.info(f'  - Hidden size: {model.config.hidden_size}')
    glog.info(f'  - Num layers: {model.config.num_hidden_layers}')
    glog.info(f'  - Num attention heads: {model.config.num_attention_heads}')
    
    if hasattr(model.config, 'num_local_experts'):
        glog.info(f'  - Num experts: {model.config.num_local_experts}')
        glog.info(f'  - Num experts per tok: {model.config.num_experts_per_tok}')
    
    # 检测模型类型
    if args.model_type == 'auto':
        args.model_type = detect_model_type(model.config)
    glog.info(f'Model type: {args.model_type}')
    
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
    all_config['model_config']['quip_params'] = quip_params
    
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(all_config, f, indent=2, default=str)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建校准数据集
    glog.info(f'Creating calibration dataset (size={args.devset_size}, ctx={args.ctx_size})...')
    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size, args.sample_proc)
    glog.info(f'Dataset shape: {devset.shape}')
    
    # 设置设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    glog.info(f'Using device: {device}')
    if torch.cuda.is_available():
        glog.info(f'GPU: {torch.cuda.get_device_name(0)}')
        glog.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # 准备嵌入
    glog.info('Computing initial embeddings...')
    with torch.no_grad():
        orig_emb = model.model.embed_tokens(devset)
    glog.info(f'Embedding shape: {orig_emb.shape}, dtype: {orig_emb.dtype}')
    
    pre_orig_emb = orig_emb.clone()
    cur_orig_emb = torch.zeros_like(orig_emb)
    
    # 准备位置ID和注意力掩码
    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        pre_orig_emb[:args.batch_size], 0)
    
    num_layers = len(model.model.layers)
    
    # 调试模式：可以只处理部分层
    start_layer = getattr(args, 'debug_start_layer', 0)
    end_layer = getattr(args, 'debug_end_layer', num_layers)
    
    glog.info(f'Processing layers {start_layer} to {end_layer-1} (total: {num_layers})')
    glog.info('='*60)
    
    for i in range(start_layer, end_layer):
        glog.info(f'\n{"="*60}')
        glog.info(f'LAYER {i}/{num_layers-1}')
        glog.info(f'{"="*60}')
        
        layer = model.model.layers[i]
        
        # 打印层结构
        if i == start_layer:
            glog.info('Layer structure:')
            for name, module in layer.named_modules():
                if len(list(module.children())) == 0:  # 叶子模块
                    if hasattr(module, 'weight'):
                        glog.info(f'  {name}: {type(module).__name__} - weight shape: {module.weight.shape}')
                    else:
                        glog.info(f'  {name}: {type(module).__name__}')
        
        # 清理内存
        utils.clean()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            glog.info(f'GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
        
        # 计算原始层输出
        if args.ft_epochs > 0:
            glog.info('Computing original layer output...')
            st = time.time()
            
            position_ids_dev = position_ids.to(device)
            attention_mask_dev = attention_mask.to(device)
            layer_dev = layer.to(device)
            
            num_batches = args.devset_size // args.batch_size
            for j in range(num_batches):
                batch_start = args.batch_size * j
                batch_end = args.batch_size * (j + 1)
                
                batch_emb = pre_orig_emb[batch_start:batch_end].to(device)
                
                with torch.no_grad():
                    output = layer_dev(
                        batch_emb,
                        position_ids=position_ids_dev,
                        attention_mask=attention_mask_dev,
                        use_cache=False,
                        output_attentions=False
                    )
                    
                    # 处理MoE输出
                    if isinstance(output, tuple):
                        layer_output = output[0]
                        if len(output) > 1 and output[1] is not None:
                            glog.debug(f'  Batch {j}: router_logits shape: {output[1].shape}')
                    else:
                        layer_output = output
                
                cur_orig_emb[batch_start:batch_end] = layer_output.cpu()
                del batch_emb, layer_output
            
            orig_msv = pre_orig_emb.float().norm()**2 / pre_orig_emb.numel()
            target_msv = cur_orig_emb.float().norm()**2 / cur_orig_emb.numel()
            
            model.model.layers[i] = layer_dev.cpu()
            del layer_dev, position_ids_dev, attention_mask_dev
            
            utils.clean()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            glog.info(f'Original output computed in {time.time()-st:.2f}s')
            glog.info(f'  Input MSV: {orig_msv:.6f}, Output MSV: {target_msv:.6f}')
        
        # 选择量化函数
        if args.model_type == 'mixtral':
            if not MIXTRAL_AVAILABLE:
                raise RuntimeError('Mixtral module not available')
            quantize_func = quantize_mixtral_layer
        else:
            if not LLAMA_AVAILABLE:
                raise RuntimeError('Llama module not available')
            quantize_func = quantize_llama_layer
        
        # 执行量化
        glog.info(f'Quantizing layer {i}...')
        st = time.time()
        
        try:
            quantize_func(
                model.model.layers[i],
                i,
                cb,
                args,
                device,
                pre_orig_emb,
                cur_orig_emb,
                model.config,
            )
            glog.info(f'Layer {i} quantized successfully in {time.time()-st:.2f}s')
            
        except Exception as e:
            glog.error(f'ERROR in layer {i}: {e}')
            import traceback
            traceback.print_exc()
            
            # 调试模式下保存错误信息
            error_info = {
                'layer': i,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            torch.save(error_info, os.path.join(args.save_path, f'error_layer{i}.pt'))
            raise
        
        # 验证量化后的输出
        if args.ft_epochs > 0 and getattr(args, 'debug_verify', False):
            glog.info('Verifying quantized layer output...')
            # 这里可以添加量化后输出的验证代码
        
        # 更新嵌入缓存
        pre_orig_emb.copy_(cur_orig_emb)
        
        # 打印内存使用情况
        if torch.cuda.is_available():
            glog.info(f'GPU memory after layer {i}: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
    
    # 保存嵌入层和输出层
    glog.info('\nSaving embedding and output layers...')
    
    torch.save({'weight': model.model.embed_tokens.weight}, 
               os.path.join(args.save_path, 'embed.pt'))
    
    torch.save({'weight': model.lm_head.weight}, 
               os.path.join(args.save_path, 'lm_head.pt'))
    
    if hasattr(model.model, 'norm'):
        torch.save({'weight': model.model.norm.weight}, 
                   os.path.join(args.save_path, 'final_norm.pt'))
    
    glog.info('='*60)
    glog.info('QUANTIZATION COMPLETED SUCCESSFULLY')
    glog.info(f'Output saved to: {args.save_path}')
    glog.info('='*60)


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
    main_debug(args)