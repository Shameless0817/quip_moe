import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from operator import attrgetter
import glog
import copy
import os

from . import quip
from .. import codebook, utils
# from ..linear import FusedLinear, QuantizedLinear, FusedQuantizedLinear
from lib.linear import *
from lib import codebook, utils


def quantize_finetune_moe_decoder_layer(
    mixed_layer, 
    quant_order, 
    idx, 
    cb, 
    args,
    device, 
    pre_orig_emb, 
    orig_emb
):
    """
    对 Mixtral 8x7B 的解码器层进行量化和微调。
    
    Mixtral 结构特点：
    - 注意力层：与 LLaMA 相同 (q_proj, k_proj, v_proj, o_proj)
    - MoE 层：包含 gate 和多个专家，每个专家有 (w1, w2, w3)
    
    Args:
        mixed_layer: 混合精度的解码器层
        quant_order: 量化顺序列表，格式为 [(attr_path, name), ...]
        idx: 层索引
        cb: codebook 对象
        args: 参数配置
        device: 计算设备
        pre_orig_emb: 原始模型前一层的嵌入
        orig_emb: 原始模型当前层的嵌入
    """
    print(quant_order)
    exit()
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)

    codebook_id = codebook.get_id(args.codebook)

    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype),
                   cb.version)
    shared_kwargs = {
        'rank': args.lora_rank,  # 恢复使用低秩修正
        'rescale_WH': args.rescale_WH,
        'resid_scale_override': args.resid_scale_override,
        'bias': False,
        'train_mode': False,  # 初始化时不删除 Hadamard 矩阵，需要用于前向传播
        'grad_ckpt': args.ft_grad_ckpt,
    }

    for quant_i, (linear_attr, name) in enumerate(quant_order):
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        
        if os.path.exists(save_path):
            glog.info(f'Skipping layer {quant_i+1}/{len(quant_order)}: {name} (already quantized)')
            
            # 加载已量化的层
            try:
                saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
                
                if saved_linear['fused']:
                    quant_linear = FusedQuantizedLinear(
                        -1, [_[0] for _ in saved_linear['shapes']],
                        saved_linear['shapes'][0][1],
                        sum([_[0] for _ in saved_linear['shapes']]), *shared_args,
                        **shared_kwargs)
                    for i in range(len(saved_linear['scales'])):
                        quant_linear.fuse_scales[i].copy_(saved_linear['scales'][i])
                else:
                    quant_linear = QuantizedLinear(saved_linear['shapes'][0][1],
                                                   saved_linear['shapes'][0][0],
                                                   *shared_args, **shared_kwargs)
                
                utils.unpack_quip(quant_linear, saved_linear, codebook_id, cb.codesz)
                
                # 验证并修复SU/SV参数
                quant_linear.SU.data = torch.clamp(quant_linear.SU.data, min=1e-8)
                quant_linear.SV.data = torch.clamp(quant_linear.SV.data, min=1e-8)
                
                quant_linear.SU = nn.Parameter(quant_linear.SU.float(), requires_grad=True)
                quant_linear.SV = nn.Parameter(quant_linear.SV.float(), requires_grad=True)
                
                split_attr = linear_attr.split('.')
                setattr(attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1], quant_linear)
            except Exception as e:
                glog.error(f'Failed to load {save_path}: {e}')
                continue
            continue
        
        glog.info(f'Quantizing layer {quant_i+1}/{len(quant_order)}: {name}')

        orig_linear = attrgetter(linear_attr)(mixed_layer)
        if orig_linear.bias is not None:
            raise Exception("Bias not implemented yet")
        
        # 根据层类型选择 Hessian 路径
        # dense 层：q, k, v, o (attention projections)
        # sparse 层：expert 相关的 w1, w2, w3
        is_dense_layer = name in ['q', 'k', 'v', 'o']
        is_expert_layer = 'expert' in name and ('_w1' in name or '_w2' in name or '_w3' in name)
        
        if hasattr(args, 'dense_hessian_path') and args.dense_hessian_path and is_dense_layer:
            hessian_base_path = args.dense_hessian_path
        elif hasattr(args, 'sparse_hessian_path') and args.sparse_hessian_path and is_expert_layer:
            hessian_base_path = args.sparse_hessian_path
        elif hasattr(args, 'hessian_path') and args.hessian_path:
            hessian_base_path = args.hessian_path
        else:
            raise ValueError(f"No Hessian path specified for layer {name}")
        
        hessian_path = f'{hessian_base_path}/{idx}_{name}.pt'
        print(f'Using Hessian path: {hessian_path}')
        
        with torch.no_grad():
            if isinstance(orig_linear, FusedLinear):
                weights = torch.split(
                    orig_linear.weight,
                    orig_linear.fuse_sizes, 
                    0
                )
            else:
                weights = [
                    orig_linear.weight
                ]
            
            quip.quantize_linear(
                weights, 
                save_path, 
                hessian_path, 
                cb, 
                args,
                device
            )

            saved_linear = torch.load(
                save_path,
                map_location=torch.device('cpu')
            )
        
            if saved_linear['fused']:
                quant_linear = FusedQuantizedLinear(
                    -1, [_[0] for _ in saved_linear['shapes']],
                    saved_linear['shapes'][0][1],
                    sum([_[0] for _ in saved_linear['shapes']]), *shared_args,
                    **shared_kwargs)
                for i in range(len(saved_linear['scales'])):
                    quant_linear.fuse_scales[i].copy_(
                        saved_linear['scales'][i])
            else:
                quant_linear = QuantizedLinear(saved_linear['shapes'][0][1],
                                                saved_linear['shapes'][0][0],
                                                *shared_args, **shared_kwargs)
            
            utils.unpack_quip(quant_linear, saved_linear, codebook_id,
                              cb.codesz)
        
        quant_linear.SU = nn.Parameter(quant_linear.SU.float(),
                                        requires_grad=True)
        quant_linear.SV = nn.Parameter(quant_linear.SV.float(),
                                        requires_grad=True)
        
        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            quant_linear)
        
        utils.clean()
        torch.cuda.empty_cache()
                
      
    if args.ft_epochs > 0:
        glog.info(f'All layers quantized, starting fine-tuning...')
        finetune_moe_decoder_layer(
            mixed_layer, 
            f'{idx}_all', 
            device,
            train_dl, 
            valid_dl, 
            args
        )
    else:
        glog.info(f'Skipping fine-tuning (ft_epochs=0)')

    # with torch.no_grad():
    #     utils.clean()
    #     for i, (linear_attr, name) in enumerate(quant_order):
    #         utils.save_susv(
    #             attrgetter(linear_attr)(mixed_layer),
    #             f'{args.save_path}/{idx}_{name}.pt')

    mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)


def extract_mixtral_susv_params(layer, quantize_gate=False):
    """
    从 Mixtral 层中提取 SU/SV 参数
    
    Returns:
        susv_params: SU/SV 参数列表
        params: 其他可训练参数列表
    """
    susv_params = []
    params = []
    
    for name, module in layer.named_modules():
        if hasattr(module, 'SU') and hasattr(module, 'SV'):
            if module.SU.requires_grad:
                susv_params.append(module.SU)
            if module.SV.requires_grad:
                susv_params.append(module.SV)
        
        if not quantize_gate and 'gate' in name:
            continue
        
        for param_name, param in module.named_parameters(recurse=False):
            if param_name not in ['SU', 'SV'] and param.requires_grad:
                # 使用 id() 来比较对象身份，避免张量比较
                param_ids = [id(p) for p in params + susv_params]
                if id(param) not in param_ids:
                    params.append(param)
    
    return susv_params, params

def get_mixtral_quant_order(num_experts=8):
    """
    获取 Mixtral 模型的量化顺序。
    
    Mixtral 解码器层结构：
    - self_attn.q_proj
    - self_attn.k_proj  
    - self_attn.v_proj
    - self_attn.o_proj
    - block_sparse_moe.gate (路由器，通常不量化)
    - block_sparse_moe.experts[i].w1 (gate projection)
    - block_sparse_moe.experts[i].w2 (down projection)
    - block_sparse_moe.experts[i].w3 (up projection)
    """
    quant_order = []
    
    # 注意力层
    quant_order.append(('self_attn.q_proj', 'q_proj'))
    quant_order.append(('self_attn.k_proj', 'k_proj'))
    quant_order.append(('self_attn.v_proj', 'v_proj'))
    quant_order.append(('self_attn.o_proj', 'o_proj'))
    
    # MoE 专家层
    for expert_idx in range(num_experts):
        quant_order.append((f'block_sparse_moe.experts.{expert_idx}.w1', 
                           f'expert{expert_idx}_w1'))
        quant_order.append((f'block_sparse_moe.experts.{expert_idx}.w2', 
                           f'expert{expert_idx}_w2'))
        quant_order.append((f'block_sparse_moe.experts.{expert_idx}.w3', 
                           f'expert{expert_idx}_w3'))
    
    return quant_order


def get_mixtral_quant_order_fused(num_experts=8):
    """
    获取 Mixtral 模型的量化顺序（融合版本）。
    假设 QKV 被融合，每个专家的 w1/w3 被融合。
    """
    quant_order = []
    
    # 注意力层 - 融合 QKV
    quant_order.append(('self_attn.qkv_proj', 'qkv_proj'))
    quant_order.append(('self_attn.o_proj', 'o_proj'))
    
    # MoE 专家层
    for expert_idx in range(num_experts):
        # w1 和 w3 通常可以融合（gate 和 up projection）
        quant_order.append((f'block_sparse_moe.experts.{expert_idx}.gate_up_proj', 
                           f'expert{expert_idx}_gate_up'))
        quant_order.append((f'block_sparse_moe.experts.{expert_idx}.w2', 
                           f'expert{expert_idx}_w2'))
    
    return quant_order


def finetune_moe_decoder_layer(
    layer, 
    name, 
    device, 
    train_dl, 
    valid_dl, 
    args,
    num_experts=8, 
    quantize_gate=False
):
    """
    对 Mixtral MoE 解码器层进行微调（简化版，与 Llama 版本保持一致）
    """
    layer = layer.to(device)
    
    # 首先禁用所有参数的梯度
    for param in layer.parameters():
        param.requires_grad = False
    
    # 只启用 QuantizedLinear 和 FusedQuantizedLinear 的 SU/SV 梯度，以及 A/B 矩阵
    for module in layer.modules():
        if isinstance(module, (QuantizedLinear, FusedQuantizedLinear)):
            if hasattr(module, 'SU'):
                module.SU.requires_grad = True
            else:
                glog.warning(f'Module {type(module).__name__} has no SU attribute')
            if hasattr(module, 'SV'):
                module.SV.requires_grad = True
            else:
                glog.warning(f'Module {type(module).__name__} has no SV attribute')
            # 也启用低秩修正参数
            if hasattr(module, 'A') and module.A is not None:
                module.A.requires_grad = True
            else:
                glog.warning(f'Module {type(module).__name__} has no A attribute')   
            if hasattr(module, 'B') and module.B is not None:
                module.B.requires_grad = True
            else:
                glog.warning(f'Module {type(module).__name__} has no B attribute')
    
    # 提取参数并创建优化器
    susv_params, params = utils.extract_susv_params(layer)
    optim = utils.get_susv_adam(susv_params, params, args)
    
    if len(susv_params) == 0 and len(params) == 0:
        glog.warning(f'layer {name} has no trainable parameters, skipping finetune')
        layer = layer.cpu()
        return
    
    glog.info(f'layer {name}: {len(susv_params)} SU/SV params, {len(params)} other params')
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 计算初始损失
    best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss}')
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    worse_ct = 0
    position_ids = None
    
    for epoch in range(args.ft_epochs):
        layer.train()
        
        for bidx, (source, targets) in enumerate(train_dl):
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            
            # 清零梯度
            optim.zero_grad()
            
            # 调试：第一个batch时检查模型状态
            if bidx == 0 and epoch == 0:
                glog.info(f'First batch - layer training={layer.training}')
                # 检查第一个可训练参数
                trainable_count = 0
                for n, p in layer.named_parameters():
                    if p.requires_grad:
                        glog.info(f'  Trainable param: {n}, shape={p.shape}')
                        trainable_count += 1
                        if trainable_count >= 3:
                            break
            
            # 注意：量化层的 codebook 操作不支持梯度，但 SU/SV 支持
            # 我们只对原始输出和目标之间的 MSE 进行优化

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = layer(source.to(device), position_ids=position_ids)[0]
                output = output.float()
                targets_f32 = targets.float().to(device)
                loss = nn.MSELoss()(output, targets_f32)
            
            scaler.scale(loss).backward()
            
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                # 定期清理显存
                if bidx % 5 == 0:
                    torch.cuda.empty_cache()
        
        # 验证
        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = calculate_mixtral_mse_loss(layer, valid_dl, device)
            
            if test_loss < best_loss:
                glog.info(f'layer {name} @ epoch {epoch} loss {test_loss:.6f} < {best_loss:.6f} BETTER')
                best_loss = test_loss
                best_sd = copy.deepcopy(layer.state_dict())
                worse_ct = 0
            else:
                glog.info(f'layer {name} @ epoch {epoch} loss {test_loss:.6f} >= {best_loss:.6f} WORSE')
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    glog.info(f'layer {name} early stopping at epoch {epoch}')
                    break
    
    del optim, train_dl, valid_dl
    layer.load_state_dict(best_sd)
    utils.clean()
    layer = layer.cpu()

def calculate_mixtral_mse_loss(layer, data_loader, device):
    """
    计算 Mixtral 层的 MSE 损失
    """
    layer.eval()
    total_loss = 0.0
    total_samples = 0
    position_ids = None
    
    with torch.no_grad():
        for source, targets in data_loader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = layer(source.to(device), position_ids=position_ids)
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = nn.MSELoss()(output, targets.to(device))
            
            total_loss += loss.item() * source.shape[0]
            total_samples += source.shape[0]
    
    return total_loss / total_samples

def quantize_mixtral_model(model, cb, args, device):
    """
    对整个 Mixtral 模型进行量化。
    """
    num_layers = len(model.model.layers)
    num_experts = model.config.num_local_experts  # 通常是 8
    
    quant_order = get_mixtral_quant_order(num_experts)
    
    # 收集每层的输入输出嵌入
    print("Collecting embeddings...")
    layer_embeddings = collect_layer_embeddings(model, args, device)
    
    for idx in range(num_layers):
        print(f"Quantizing layer {idx}/{num_layers}...")
        
        mixed_layer = model.model.layers[idx]
        pre_orig_emb = layer_embeddings[idx]['input']
        orig_emb = layer_embeddings[idx]['output']
        
        quantize_finetune_moe_decoder_layer(
            mixed_layer, quant_order, idx, cb, args,
            device, pre_orig_emb, orig_emb)
        
        # 更新嵌入用于下一层
        if idx < num_layers - 1:
            with torch.no_grad():
                layer_embeddings[idx + 1]['input'] = compute_layer_output(
                    model.model.layers[idx], layer_embeddings[idx]['input'], device)
        
        utils.clean()
    
    print("Quantization complete!")
    return model


def collect_layer_embeddings(model, args, device):
    """
    收集每层的输入输出嵌入用于微调。
    """
    layer_embeddings = {}
    
    # 这里需要根据实际的校准数据集实现
    # 简化版本：使用 hook 收集
    
    hooks = []
    
    def make_hook(layer_idx, is_input):
        def hook(module, input, output):
            if is_input:
                layer_embeddings.setdefault(layer_idx, {})['input'] = input[0].detach().cpu()
            else:
                if isinstance(output, tuple):
                    output = output[0]
                layer_embeddings.setdefault(layer_idx, {})['output'] = output.detach().cpu()
        return hook
    
    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(idx, is_input=False)))
        hooks.append(layer.register_forward_pre_hook(
            lambda m, i, idx=idx: layer_embeddings.setdefault(idx, {}).__setitem__('input', i[0].detach().cpu())))
    
    # 运行校准数据
    model.eval()
    with torch.no_grad():
        for batch in args.calib_loader:
            input_ids = batch['input_ids'].to(device)
            model(input_ids)
            break  # 只需要一个 batch
    
    for hook in hooks:
        hook.remove()
    
    return layer_embeddings