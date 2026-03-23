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
from lib.linear import *
from lib import codebook, utils


def quantize_finetune_qwen_decoder_layer(
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
    对 Qwen1.5 MoE 的解码器层进行量化和微调。
    
    Qwen1.5 MoE 结构特点：
    - 注意力层：q_proj, k_proj, v_proj (有bias), o_proj (无bias)
    - MoE 层：
        - gate (路由器)
        - 60个路由专家，每个专家有 (gate_proj, up_proj, down_proj)
        - 1个共享专家 (shared_expert) + shared_expert_gate
    
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
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)

    codebook_id = codebook.get_id(args.codebook)

    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype),
                   cb.version)
    shared_kwargs = {
        'rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'resid_scale_override': args.resid_scale_override,
        'train_mode': False,
        'grad_ckpt': args.ft_grad_ckpt,
    }

    # 分批处理专家以节省内存
    expert_batch_size = getattr(args, 'expert_batch_size', 4)
    
    for quant_i, (linear_attr, name) in enumerate(quant_order):
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        
        if os.path.exists(save_path):
            glog.info(f'Skipping layer {quant_i+1}/{len(quant_order)}: {name} (already quantized)')
            
            # 加载已量化的层
            try:
                saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
                
                # 判断是否有 bias (Qwen 的 attention qkv 有 bias)
                has_bias = saved_linear.get('bias', None) is not None
                shared_kwargs['bias'] = has_bias
                
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
                
                # 恢复 shared_kwargs 中的 bias 设置
                shared_kwargs['bias'] = False
                
            except Exception as e:
                glog.error(f'Failed to load {save_path}: {e}')
                continue
            continue
        
        glog.info(f'Quantizing layer {quant_i+1}/{len(quant_order)}: {name}')

        orig_linear = attrgetter(linear_attr)(mixed_layer)
        
        # Qwen 的 attention 层有 bias，需要特殊处理
        has_bias = orig_linear.bias is not None
        if has_bias:
            glog.info(f'Layer {name} has bias, will be saved separately')
        
        # 根据层类型选择 Hessian 路径
        # dense 层：q, k, v, o (attention projections)
        # sparse 层：expert 相关的 gate_proj, up_proj, down_proj
        # shared 层：shared_expert 相关
        is_dense_layer = name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        is_expert_layer = 'expert' in name and ('gate_proj' in name or 'up_proj' in name or 'down_proj' in name)
        is_shared_expert = 'shared_expert' in name
        is_gate = name == 'gate'
        
        if hasattr(args, 'dense_hessian_path') and args.dense_hessian_path and is_dense_layer:
            hessian_base_path = args.dense_hessian_path
        elif hasattr(args, 'sparse_hessian_path') and args.sparse_hessian_path and is_expert_layer:
            hessian_base_path = args.sparse_hessian_path
        elif hasattr(args, 'shared_hessian_path') and args.shared_hessian_path and is_shared_expert:
            hessian_base_path = args.shared_hessian_path
        elif hasattr(args, 'hessian_path') and args.hessian_path:
            hessian_base_path = args.hessian_path
        else:
            raise ValueError(f"No Hessian path specified for layer {name}")
        
        # 简化名称以匹配 Hessian 文件命名（例如 q_proj -> q）
        hessian_name = name.replace('_proj', '') if is_dense_layer else name
        hessian_path = f'{hessian_base_path}/{idx}_{hessian_name}.pt'
        glog.info(f'Using Hessian path: {hessian_path}')
        
        # 检查是否要跳过门控网络量化
        if is_gate and not getattr(args, 'quantize_gate', False):
            glog.info(f'Skipping gate quantization (quantize_gate=False)')
            # 保存原始权重
            gate_state = {
                'weight': orig_linear.weight.data.clone().half(),
                'bias': orig_linear.bias.data.clone().half() if orig_linear.bias is not None else None,
                'quantized': False,
            }
            torch.save(gate_state, save_path)
            continue
        
        # 检查是否要跳过共享专家量化
        if is_shared_expert and not getattr(args, 'quantize_shared_expert', True):
            glog.info(f'Skipping shared expert quantization (quantize_shared_expert=False)')
            shared_state = {
                'weight': orig_linear.weight.data.clone().half(),
                'bias': orig_linear.bias.data.clone().half() if orig_linear.bias is not None else None,
                'quantized': False,
            }
            torch.save(shared_state, save_path)
            continue
        
        with torch.no_grad():
            if isinstance(orig_linear, FusedLinear):
                weights = torch.split(
                    orig_linear.weight,
                    orig_linear.fuse_sizes, 
                    0
                )
            else:
                weights = [orig_linear.weight]
            
            # 量化权重
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
            
            # 保存 bias（如果存在）
            if has_bias:
                saved_linear['bias'] = orig_linear.bias.data.clone()
                torch.save(saved_linear, save_path)
        
            # 创建量化层
            shared_kwargs['bias'] = has_bias
            
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
            
            # 恢复 bias
            if has_bias and 'bias' in saved_linear:
                quant_linear.bias.data.copy_(saved_linear['bias'])
        
        quant_linear.SU = nn.Parameter(quant_linear.SU.float(),
                                        requires_grad=True)
        quant_linear.SV = nn.Parameter(quant_linear.SV.float(),
                                        requires_grad=True)
        
        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            quant_linear)
        
        # 恢复 shared_kwargs 中的 bias 设置
        shared_kwargs['bias'] = False
        
        utils.clean()
        torch.cuda.empty_cache()
        
        # 定期清理内存（特别是处理专家时）
        if is_expert_layer and (quant_i % expert_batch_size == 0):
            glog.info(f'Cleaning up memory after processing {expert_batch_size} experts')
            utils.clean()
            torch.cuda.empty_cache()
                
      
    if args.ft_epochs > 0:
        glog.info(f'All layers quantized, starting fine-tuning...')
        finetune_qwen_decoder_layer(
            mixed_layer, 
            f'{idx}_all', 
            device,
            train_dl, 
            valid_dl, 
            args
        )
    else:
        glog.info(f'Skipping fine-tuning (ft_epochs=0)')

    mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)


def extract_qwen_susv_params(layer, quantize_gate=False, quantize_shared_expert=True):
    """
    从 Qwen1.5 MoE 层中提取 SU/SV 参数
    
    Args:
        layer: Qwen decoder layer
        quantize_gate: 是否量化了门控网络
        quantize_shared_expert: 是否量化了共享专家
    
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
        
        # 跳过未量化的组件
        if not quantize_gate and 'mlp.gate' in name:
            continue
        
        if not quantize_shared_expert and 'shared_expert' in name:
            continue
        
        for param_name, param in module.named_parameters(recurse=False):
            if param_name not in ['SU', 'SV'] and param.requires_grad:
                param_ids = [id(p) for p in params + susv_params]
                if id(param) not in param_ids:
                    params.append(param)
    
    return susv_params, params


def get_qwen_quant_order(num_routed_experts=60, num_shared_experts=1, 
                         quantize_gate=False, quantize_shared_expert=True):
    """
    获取 Qwen1.5 MoE 模型的量化顺序。
    
    Qwen1.5 MoE 解码器层结构：
    - self_attn.q_proj (有 bias)
    - self_attn.k_proj (有 bias)
    - self_attn.v_proj (有 bias)
    - self_attn.o_proj (无 bias)
    - mlp.gate (路由器)
    - mlp.experts[i].gate_proj (gate projection)
    - mlp.experts[i].up_proj (up projection)
    - mlp.experts[i].down_proj (down projection)
    - mlp.shared_expert.gate_proj (共享专家)
    - mlp.shared_expert.up_proj
    - mlp.shared_expert.down_proj
    - mlp.shared_expert_gate (共享专家门控)
    
    Args:
        num_routed_experts: 路由专家数量（默认60）
        num_shared_experts: 共享专家数量（默认1）
        quantize_gate: 是否量化门控网络
        quantize_shared_expert: 是否量化共享专家
    """
    quant_order = []
    
    # 注意力层
    quant_order.append(('self_attn.q_proj', 'q_proj'))
    quant_order.append(('self_attn.k_proj', 'k_proj'))
    quant_order.append(('self_attn.v_proj', 'v_proj'))
    quant_order.append(('self_attn.o_proj', 'o_proj'))
    
    # 门控网络（可选）
    if quantize_gate:
        quant_order.append(('mlp.gate', 'gate'))
    
    # 路由专家层
    for expert_idx in range(num_routed_experts):
        quant_order.append((f'mlp.experts.{expert_idx}.gate_proj', 
                           f'expert{expert_idx}_gate_proj'))
        quant_order.append((f'mlp.experts.{expert_idx}.up_proj', 
                           f'expert{expert_idx}_up_proj'))
        quant_order.append((f'mlp.experts.{expert_idx}.down_proj', 
                           f'expert{expert_idx}_down_proj'))
    
    # 共享专家（可选）
    if num_shared_experts > 0 and quantize_shared_expert:
        quant_order.append(('mlp.shared_expert.gate_proj', 
                           'shared_expert_gate_proj'))
        quant_order.append(('mlp.shared_expert.up_proj', 
                           'shared_expert_up_proj'))
        quant_order.append(('mlp.shared_expert.down_proj', 
                           'shared_expert_down_proj'))
        quant_order.append(('mlp.shared_expert_gate', 
                           'shared_expert_gate'))
    
    return quant_order


def get_qwen_quant_order_fused(num_routed_experts=60, num_shared_experts=1,
                                quantize_gate=False, quantize_shared_expert=True):
    """
    获取 Qwen1.5 MoE 模型的量化顺序（融合版本）。
    假设 QKV 被融合，每个专家的 gate_proj/up_proj 被融合。
    
    注意：Qwen 的 QKV 有 bias，融合时需要特别处理
    """
    quant_order = []
    
    # 注意力层 - 融合 QKV（注意：有 bias）
    quant_order.append(('self_attn.qkv_proj', 'qkv_proj'))
    quant_order.append(('self_attn.o_proj', 'o_proj'))
    
    # 门控网络（可选）
    if quantize_gate:
        quant_order.append(('mlp.gate', 'gate'))
    
    # 路由专家层 - 融合 gate_proj 和 up_proj
    for expert_idx in range(num_routed_experts):
        quant_order.append((f'mlp.experts.{expert_idx}.gate_up_proj', 
                           f'expert{expert_idx}_gate_up'))
        quant_order.append((f'mlp.experts.{expert_idx}.down_proj', 
                           f'expert{expert_idx}_down_proj'))
    
    # 共享专家（可选）
    if num_shared_experts > 0 and quantize_shared_expert:
        quant_order.append(('mlp.shared_expert.gate_up_proj', 
                           'shared_expert_gate_up'))
        quant_order.append(('mlp.shared_expert.down_proj', 
                           'shared_expert_down_proj'))
        quant_order.append(('mlp.shared_expert_gate', 
                           'shared_expert_gate'))
    
    return quant_order


def finetune_qwen_decoder_layer(
    layer, 
    name, 
    device, 
    train_dl, 
    valid_dl, 
    args,
    num_routed_experts=60,
    num_shared_experts=1,
    quantize_gate=False,
    quantize_shared_expert=True
):
    """
    对 Qwen1.5 MoE 解码器层进行微调
    
    Args:
        layer: Qwen decoder layer
        name: 层名称
        device: 计算设备
        train_dl: 训练数据加载器
        valid_dl: 验证数据加载器
        args: 参数配置
        num_routed_experts: 路由专家数量
        num_shared_experts: 共享专家数量
        quantize_gate: 是否量化了门控网络
        quantize_shared_expert: 是否量化了共享专家
    """
    layer = layer.to(device)
    
    # 首先禁用所有参数的梯度
    for param in layer.parameters():
        param.requires_grad = False
    
    # 只启用 QuantizedLinear 和 FusedQuantizedLinear 的 SU/SV 梯度，以及 A/B 矩阵
    trainable_modules = 0
    for name_module, module in layer.named_modules():
        if isinstance(module, (QuantizedLinear, FusedQuantizedLinear)):
            if hasattr(module, 'SU') and module.SU is not None:
                module.SU.requires_grad = True
                trainable_modules += 1
            else:
                glog.warning(f'Module {name_module} has no SU attribute')
            
            if hasattr(module, 'SV') and module.SV is not None:
                module.SV.requires_grad = True
            else:
                glog.warning(f'Module {name_module} has no SV attribute')
            
            # 启用低秩修正参数
            if hasattr(module, 'A') and module.A is not None:
                module.A.requires_grad = True
            
            if hasattr(module, 'B') and module.B is not None:
                module.B.requires_grad = True
    
    glog.info(f'Found {trainable_modules} trainable quantized modules')
    
    # 提取参数并创建优化器
    susv_params, params = extract_qwen_susv_params(
        layer, 
        quantize_gate=quantize_gate,
        quantize_shared_expert=quantize_shared_expert
    )
    
    if len(susv_params) == 0 and len(params) == 0:
        glog.warning(f'layer {name} has no trainable parameters, skipping finetune')
        layer = layer.cpu()
        return
    
    glog.info(f'layer {name}: {len(susv_params)} SU/SV params, {len(params)} other params')
    
    # 创建优化器
    optim = utils.get_susv_adam(susv_params, params, args)
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 计算初始损失
    best_loss = calculate_qwen_mse_loss(layer, valid_dl, device)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss:.6f}')
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    worse_ct = 0
    position_ids = None
    
    for epoch in range(args.ft_epochs):
        layer.train() if args.ft_train_mode else layer.eval()
        
        for bidx, (source, targets) in enumerate(train_dl):
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            
            # 清零梯度
            optim.zero_grad()
            
            # 调试：第一个batch时检查模型状态
            if bidx == 0 and epoch == 0:
                glog.info(f'First batch - layer training={layer.training}')
                trainable_count = 0
                for n, p in layer.named_parameters():
                    if p.requires_grad:
                        glog.info(f'  Trainable param: {n}, shape={p.shape}, dtype={p.dtype}')
                        trainable_count += 1
                        if trainable_count >= 5:
                            break
                
                if trainable_count == 0:
                    glog.error('No trainable parameters found!')
                    break
            
            # 前向传播
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = layer(source.to(device), position_ids=position_ids)
                if isinstance(output, tuple):
                    output = output[0]
                
                output = output.float()
                targets_f32 = targets.float().to(device)
                loss = nn.MSELoss()(output, targets_f32)
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                
                # 定期清理显存
                if bidx % 5 == 0:
                    torch.cuda.empty_cache()
        
        # 验证
        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = calculate_qwen_mse_loss(layer, valid_dl, device)
            
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


def calculate_qwen_mse_loss(layer, data_loader, device):
    """
    计算 Qwen1.5 MoE 层的 MSE 损失
    
    Args:
        layer: Qwen decoder layer
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        float: 平均 MSE 损失
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
                
                output = output.float()
                targets_f32 = targets.float().to(device)
                loss = nn.MSELoss()(output, targets_f32)
            
            total_loss += loss.item() * source.shape[0]
            total_samples += source.shape[0]
    
    return total_loss / total_samples


def quantize_qwen_model(model, cb, args, device):
    """
    对整个 Qwen1.5 MoE 模型进行量化。
    
    Args:
        model: Qwen1.5 MoE 模型
        cb: codebook 对象
        args: 参数配置
        device: 计算设备
    
    Returns:
        model: 量化后的模型
    """
    num_layers = len(model.model.layers)
    num_routed_experts = model.config.n_routed_experts  # 通常是 60
    num_shared_experts = getattr(model.config, 'num_shared_experts', 1)  # 通常是 1
    
    quantize_gate = getattr(args, 'quantize_gate', False)
    quantize_shared_expert = getattr(args, 'quantize_shared_expert', True)
    
    glog.info(f'Quantizing Qwen1.5 MoE model:')
    glog.info(f'  - Total layers: {num_layers}')
    glog.info(f'  - Routed experts per layer: {num_routed_experts}')
    glog.info(f'  - Shared experts per layer: {num_shared_experts}')
    glog.info(f'  - Quantize gate: {quantize_gate}')
    glog.info(f'  - Quantize shared expert: {quantize_shared_expert}')
    
    quant_order = get_qwen_quant_order(
        num_routed_experts=num_routed_experts,
        num_shared_experts=num_shared_experts,
        quantize_gate=quantize_gate,
        quantize_shared_expert=quantize_shared_expert
    )
    
    glog.info(f'Quantization order: {len(quant_order)} layers per decoder layer')
    
    # 收集每层的输入输出嵌入
    glog.info("Collecting embeddings...")
    layer_embeddings = collect_qwen_layer_embeddings(model, args, device)
    
    for idx in range(num_layers):
        glog.info(f"=" * 80)
        glog.info(f"Quantizing layer {idx+1}/{num_layers}...")
        
        mixed_layer = model.model.layers[idx]
        pre_orig_emb = layer_embeddings[idx]['input']
        orig_emb = layer_embeddings[idx]['output']
        
        quantize_finetune_qwen_decoder_layer(
            mixed_layer, quant_order, idx, cb, args,
            device, pre_orig_emb, orig_emb)
        
        # 更新嵌入用于下一层
        if idx < num_layers - 1:
            with torch.no_grad():
                layer_embeddings[idx + 1]['input'] = compute_qwen_layer_output(
                    model.model.layers[idx], layer_embeddings[idx]['input'], device)
        
        utils.clean()
        torch.cuda.empty_cache()
    
    glog.info("=" * 80)
    glog.info("Quantization complete!")
    return model


def collect_qwen_layer_embeddings(model, args, device):
    """
    收集 Qwen1.5 MoE 每层的输入输出嵌入用于微调。
    
    Args:
        model: Qwen1.5 MoE 模型
        args: 参数配置
        device: 计算设备
    
    Returns:
        dict: 每层的输入输出嵌入
    """
    layer_embeddings = {}
    
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
    
    # 注册 hooks
    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(idx, is_input=False)))
        hooks.append(layer.register_forward_pre_hook(
            lambda m, i, idx=idx: layer_embeddings.setdefault(idx, {}).__setitem__('input', i[0].detach().cpu())))
    
    # 运行校准数据
    model.eval()
    with torch.no_grad():
        if hasattr(args, 'calib_loader') and args.calib_loader is not None:
            for batch in args.calib_loader:
                input_ids = batch['input_ids'].to(device)
                model(input_ids)
                break  # 只需要一个 batch
        else:
            glog.warning('No calibration loader provided, using random data')
            # 使用随机数据
            random_input = torch.randint(0, model.config.vocab_size, 
                                        (args.batch_size, args.ctx_size),
                                        device=device)
            model(random_input)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    glog.info(f'Collected embeddings for {len(layer_embeddings)} layers')
    return layer_embeddings


def compute_qwen_layer_output(layer, input_emb, device):
    """
    计算 Qwen1.5 MoE 层的输出
    
    Args:
        layer: Qwen decoder layer
        input_emb: 输入嵌入
        device: 计算设备
    
    Returns:
        torch.Tensor: 输出嵌入
    """
    layer = layer.to(device)
    input_emb = input_emb.to(device)
    
    position_ids = torch.arange(input_emb.shape[1], device=device).unsqueeze(0)
    
    with torch.no_grad():
        output = layer(input_emb, position_ids=position_ids)
        if isinstance(output, tuple):
            output = output[0]
    
    layer = layer.cpu()
    return output.cpu()