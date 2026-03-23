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
    对 DeepSeek MoE 的解码器层进行量化和微调。
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
        'bias': False,
        'train_mode': False,
        'grad_ckpt': args.ft_grad_ckpt,
    }

    for quant_i, (linear_attr, name) in enumerate(quant_order):
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        
        if os.path.exists(save_path):
            glog.info(f'Skipping layer {quant_i+1}/{len(quant_order)}: {name} (already quantized)')
            
            saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
            
            # 调试信息
            glog.info(f'Loading {name}: shapes={saved_linear["shapes"]}, fused={saved_linear["fused"]}')
            glog.info(f'saved_linear keys: {saved_linear.keys()}')
            glog.info(f'SU shape in saved: {saved_linear["SU"].shape}, SV shape: {saved_linear["SV"].shape}')
            
            if saved_linear['fused']:
                quant_linear = FusedQuantizedLinear(
                    -1, [_[0] for _ in saved_linear['shapes']],
                    saved_linear['shapes'][0][1],
                    sum([_[0] for _ in saved_linear['shapes']]), *shared_args,
                    **shared_kwargs)
                for i in range(len(saved_linear['scales'])):
                    quant_linear.fuse_scales[i].copy_(saved_linear['scales'][i])
            else:
                in_features = saved_linear['shapes'][0][1]
                out_features = saved_linear['shapes'][0][0]
                glog.info(f'Creating QuantizedLinear: in_features={in_features}, out_features={out_features}')
                quant_linear = QuantizedLinear(
                    in_features, 
                    out_features,
                    *shared_args, 
                    **shared_kwargs
                )
                glog.info(f'Created QuantizedLinear: SU shape={quant_linear.SU.shape}, SV shape={quant_linear.SV.shape}')

            # Old cache may come from deprecated settings and have incompatible
            # SU/SV shapes (e.g. matrix SU from removed kron path).
            if ('SU' in saved_linear and 'SV' in saved_linear and
                (saved_linear['SU'].shape != quant_linear.SU.shape or
                 saved_linear['SV'].shape != quant_linear.SV.shape)):
                glog.warning(
                    f'Cached file {save_path} has incompatible SU/SV shapes: '
                    f"saved SU={tuple(saved_linear['SU'].shape)}, SV={tuple(saved_linear['SV'].shape)}; "
                    f"expected SU={tuple(quant_linear.SU.shape)}, SV={tuple(quant_linear.SV.shape)}. "
                    'Deleting stale cache and re-quantizing this layer.'
                )
                os.remove(save_path)
            else:
                utils.unpack_quip(quant_linear, saved_linear, codebook_id, cb.codesz)
                # Do not clamp SU/SV. They encode signed/orthogonal transforms and
                # clamping destroys the transform, causing large activation errors.
                
                quant_linear.SU = nn.Parameter(quant_linear.SU.float(), requires_grad=True)
                quant_linear.SV = nn.Parameter(quant_linear.SV.float(), requires_grad=True)
                
                split_attr = linear_attr.split('.')
                setattr(attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1], quant_linear)
                
                # 跳过后续的量化步骤，因为该层已经量化完成
                continue
        
        # 只有当文件不存在时才进行量化
        glog.info(f'Quantizing layer {quant_i+1}/{len(quant_order)}: {name}')

        orig_linear = attrgetter(linear_attr)(mixed_layer)
        if orig_linear.bias is not None:
            raise Exception("Bias not implemented yet")
        
        # dense_hessian_path -> QKVO 注意力层 (hessians_deepseek_moe_16b_base_qkvo)
        # sparse_hessian_path -> 所有 MLP 层 (hessians_deepseek_moe_16b_base_experts)

        is_attention_layer = name in ['q', 'k', 'v', 'o', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
        is_mlp_layer = name.startswith('dense_') or name.startswith('shared_') or name.startswith('expert')
        
        if hasattr(args, 'dense_hessian_path') and args.dense_hessian_path and is_attention_layer:
            hessian_base_path = args.dense_hessian_path
        elif hasattr(args, 'sparse_hessian_path') and args.sparse_hessian_path and is_mlp_layer:
            hessian_base_path = args.sparse_hessian_path
        elif hasattr(args, 'hessian_path') and args.hessian_path:
            hessian_base_path = args.hessian_path
        else:
            raise ValueError(f"No Hessian path specified for layer {name}")
        
        # 只对注意力层移除 _proj 后缀 (QKVO 文件名: 0_q.pt, 0_k.pt, 0_v.pt, 0_o.pt)
        # MLP 层保留完整名称 (Expert 文件名: 0_dense_gate_proj.pt, 1_shared_gate_proj.pt, 等)
        if is_attention_layer:
            hessian_name = name.replace('_proj', '')
        else:
            hessian_name = name
        
        hessian_path = f'{hessian_base_path}/{idx}_{hessian_name}.pt'
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
            
            utils.unpack_quip(
                quant_linear, 
                saved_linear, 
                codebook_id,
                cb.codesz
            )
        
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

    mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)


def extract_deepseek_susv_params(layer, quantize_gate=False):
    """
    从 DeepSeek 层中提取 SU/SV 参数
    """
    susv_params = []
    params = []
    
    for name, module in layer.named_modules():
        if hasattr(module, 'SU') and hasattr(module, 'SV'):
            if module.SU.requires_grad:
                susv_params.append(module.SU)
            if module.SV.requires_grad:
                susv_params.append(module.SV)
        
        if not quantize_gate and 'gate' in name and 'proj' not in name:
            # 排除路由器 gate，但保留 gate_proj
            continue
        
        for param_name, param in module.named_parameters(recurse=False):
            if param_name not in ['SU', 'SV'] and param.requires_grad:
                param_ids = [id(p) for p in params + susv_params]
                if id(param) not in param_ids:
                    params.append(param)
    
    return susv_params, params


def get_deepseek_quant_order(layer):
    """
    动态获取 DeepSeek 模型的量化顺序。
    支持 DeepSeek 的 Dense 层和 MoE 层(包含 Shared Experts 和 Routed Experts)。
    """
    quant_order = []
    
    # 注意力层
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if hasattr(layer.self_attn, proj):
            quant_order.append((f'self_attn.{proj}', proj[0]))
    
    # MLP / MoE 层
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        
    # Dense layer
        if hasattr(mlp, 'gate_proj') and not hasattr(mlp, 'experts'):
            quant_order.append(('mlp.gate_proj', 'dense_gate_proj'))
            quant_order.append(('mlp.up_proj', 'dense_up_proj'))
            quant_order.append(('mlp.down_proj', 'dense_down_proj'))
            
        # 2. MoE 层
        elif hasattr(mlp, 'experts'):
            # 共享专家 (Shared Experts) - 名称需与 hessian_offline_deepseek_experts.py 一致
            if hasattr(mlp, 'shared_experts') and mlp.shared_experts is not None:
                quant_order.append(('mlp.shared_experts.gate_proj', 'shared_gate_proj'))
                quant_order.append(('mlp.shared_experts.up_proj', 'shared_up_proj'))
                quant_order.append(('mlp.shared_experts.down_proj', 'shared_down_proj'))
            
            # 路由专家 (Routed Experts)
            for i, expert in enumerate(mlp.experts):
                quant_order.append((f'mlp.experts.{i}.gate_proj', f'expert{i}_gate_proj'))
                quant_order.append((f'mlp.experts.{i}.up_proj', f'expert{i}_up_proj'))
                quant_order.append((f'mlp.experts.{i}.down_proj', f'expert{i}_down_proj'))
    
    return quant_order


def finetune_moe_decoder_layer(
    layer, 
    name, 
    device, 
    train_dl, 
    valid_dl, 
    args,
    quantize_gate=False
):
    """
    对 DeepSeek MoE 解码器层进行微调
    """
    layer = layer.to(device)
    
    for param in layer.parameters():
        param.requires_grad = False
    
    for module in layer.modules():
        if isinstance(module, (QuantizedLinear, FusedQuantizedLinear)):
            if hasattr(module, 'SU'):
                module.SU.requires_grad = True
            if hasattr(module, 'SV'):
                module.SV.requires_grad = True
            if hasattr(module, 'A') and module.A is not None:
                module.A.requires_grad = True
            if hasattr(module, 'B') and module.B is not None:
                module.B.requires_grad = True
    
    susv_params, params = extract_deepseek_susv_params(layer, quantize_gate)
    optim = utils.get_susv_adam(susv_params, params, args)
    
    if len(susv_params) == 0 and len(params) == 0:
        glog.warning(f'layer {name} has no trainable parameters, skipping finetune')
        layer = layer.cpu()
        return
    
    glog.info(f'layer {name}: {len(susv_params)} SU/SV params, {len(params)} other params')
    
    torch.cuda.empty_cache()
    
    best_loss = calculate_deepseek_mse_loss(layer, valid_dl, device)
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
            
            optim.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = layer(source.to(device), position_ids=position_ids)[0]
                output = output.float()
                targets_f32 = targets.float().to(device)
                loss = nn.MSELoss()(output, targets_f32)
            
            scaler.scale(loss).backward()
            
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                if bidx % 5 == 0:
                    torch.cuda.empty_cache()
        
        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = calculate_deepseek_mse_loss(layer, valid_dl, device)
            
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


def calculate_deepseek_mse_loss(layer, data_loader, device):
    """
    计算 DeepSeek 层的 MSE 损失
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


def quantize_deepseek_model(model, cb, args, device):
    """
    对整个 DeepSeek 模型进行量化。
    """
    num_layers = len(model.model.layers)
    
    print("Collecting embeddings...")
    layer_embeddings = collect_layer_embeddings(model, args, device)
    
    for idx in range(num_layers):
        print(f"Quantizing layer {idx}/{num_layers}...")
        
        mixed_layer = model.model.layers[idx]
        pre_orig_emb = layer_embeddings[idx]['input']
        orig_emb = layer_embeddings[idx]['output']
        
        # 动态获取当前层的量化顺序 (自动区分 Dense 层和 MoE 层)
        quant_order = get_deepseek_quant_order(mixed_layer)
        
        quantize_finetune_moe_decoder_layer(
            mixed_layer, quant_order, idx, cb, args,
            device, pre_orig_emb, orig_emb)
        
        if idx < num_layers - 1:
            with torch.no_grad():
                # 注意：这里可能需要根据您的 compute_layer_output 实现传入 position_ids
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
    
    model.eval()
    with torch.no_grad():
        for batch in args.calib_loader:
            input_ids = batch['input_ids'].to(device)
            model(input_ids)
            break 
    
    for hook in hooks:
        hook.remove()
    
    return layer_embeddings

# 假设外部有 compute_layer_output 的实现，这里仅作占位说明
def compute_layer_output(layer, inputs, device):
    layer.eval()
    position_ids = torch.arange(inputs.shape[1], device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = layer(inputs.to(device), position_ids=position_ids)
    if isinstance(outputs, tuple):
        return outputs[0].cpu()
    return outputs.cpu()