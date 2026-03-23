'''
将已经量化好的mixtral模型转化成带lora补偿的形式
'''

import torch
import time
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from lib.linear import QuantizedLinear

from lib.codebook.latticee8_padded12 import (
    QuantizedE8P12LinearWithLoRA,
    add_lora_to_e8p_layer as _add_lora_to_e8p_layer_orig,
    compute_quantization_error_e8p
)


# def add_lora_to_e8p_layer(
#     original_module,
#     original_weight,
#     rank=16,
#     Qidxs_list=None,
#     SU=None,
#     SV=None,
#     had_left=None,
#     had_right=None,
#     K_left=None,
#     K_right=None,
#     bias=None
# ):
#     from lib.codebook.latticee8_padded12 import E8P12_codebook
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
         
    
#     new_module = QuantizedE8P12LinearWithLoRA(
#         original_module=original_module,
#         lora_rank=rank,
#         device=device 
#     )
#     # 缓存量化参数
#     if Qidxs_list is not None:  
#         new_module.cache_params(
#           Qidxs_list, SU, SV, had_left, had_right, K_left, K_right, bias)
#     else: 
#         raise NotImplementedError("需要提供量化参数")
#     # 计算量化误差
#     d_in = len(new_module.SU)
#     d_out = len(new_module.SV) 
#     W_decompressed = new_module.decompress_weights() 
    

def add_lora_to_e8p_layer(
    original_module, 
    original_weight, 
    rank=16, 
    Qidxs_list=None, 
    SU=None, 
    SV=None, 
    had_left=None, 
    had_right=None,
    K_left=None, 
    K_right=None, 
    bias=None
):
    """为单个E8P层添加LoRA补偿（修复版本）
    
    这个函数为 QuantizedLinear 添加 LoRA 补偿。
    """
    from lib.linear.quantized_linear_withlora import QuantizedLinearWithlora
     
    # 安全地获取设备 - 尝试从 parameters、buffers 或使用默认值
    device = None
    try:
        device = next(original_module.parameters()).device
    except StopIteration:
        # 如果没有 parameters，尝试从 buffers 获取
        try:
            device = next(original_module.buffers()).device
        except StopIteration:
            # 如果都没有，使用 CUDA（如果可用）或 CPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"    ⚠ 无法从模块获取设备，使用默认设备: {device}")

    # initialization
    from lib.linear.quantized_linear_withlora import convert_to_lora_linear 
    
    new_module = convert_to_lora_linear(
        original_module,
        rank, 
    )
    
    # 在调用 decompress_weights 之前，保存 Qidxs 的副本
    # 因为 decompress_weights 会删除它并创建 Qidxs_0, Qidxs_1 等
    qidxs_backup = None
    if hasattr(new_module, 'Qidxs') and new_module.Qidxs is not None:
        qidxs_backup = new_module.Qidxs.clone()
        print(f"    保存 Qidxs 备份: shape={qidxs_backup.shape}, dtype={qidxs_backup.dtype}")

    # delta_w 
    print(f"    原始权重形状: {original_weight.shape}")
    
    # 解压权重需要重塑到正确的维度
    d_in = len(new_module.SU)
    d_out = len(new_module.SV)
    print(f"    d_in={d_in}, d_out={d_out}")
    
    # 获取解压后的权重
    W_decompressed = new_module.decompress_weights()
    print(f"    解压权重形状: {W_decompressed.shape}")
    
    # 确保原始权重在同一设备
    original_weight_device = original_weight.to(W_decompressed.device)
    
    # 简单的误差计算
    delta_w = original_weight_device - W_decompressed.float()
    print(f"    误差矩阵形状: {delta_w.shape}")
    
    # SVD分解
    print("    执行SVD分解...")
    new_module.initialize_lora_from_decomposition(delta_w)
    print(f"    LoRA初始化完成")
    
    # 恢复原始的 Qidxs（如果有备份）
    # 移除 unpacked 版本 (Qidxs_0, Qidxs_1, ...)
    if qidxs_backup is not None:
        i = 0
        while hasattr(new_module, f'Qidxs_{i}'):
            delattr(new_module, f'Qidxs_{i}')
            i += 1
        if i > 0:
            print(f"    移除了 {i} 个 unpacked Qidxs")
        
        # 恢复原始 Qidxs
        new_module.register_buffer('Qidxs', qidxs_backup, persistent=True)
        # 重置 built_codebook_class 标志，以便下次使用时重新 unpack
        new_module.built_codebook_class = False
        print(f"    恢复 Qidxs: shape={qidxs_backup.shape}")
    
    return new_module


def load_quantized_mixtral_model(
    quantized_path,
    device_map="auto",
    torch_dtype=torch.float16,
    model_class=None,
    verbose=True
):
    """
    根据hfize_mixtral的加载方式，加载量化的Mixtral模型。
    
    该函数完全模拟hfize_mixtral中的模型加载流程，包括：
    1. 加载模型配置和量化参数
    2. 创建模型实例（带有QuantizedLinear层）
    3. 加载所有量化的权重（embeddings, LayerNorms, 注意力层, 专家层等）
    4. 返回完整的可推理模型
    
    Args:
        quantized_path (str): 量化模型保存的路径（包含config.pt和各层权重文件）
        device_map (str, optional): 设备映射策略，默认"auto"自动分配到GPU/CPU
        torch_dtype (torch.dtype, optional): 模型的数据类型，默认torch.float16
        model_class (class, optional): 自定义的模型类，默认使用MixtralForCausalLM
        verbose (bool, optional): 是否打印加载进度信息，默认True
        
    Returns:
        tuple: (model, config) - 加载完整的模型和其配置对象
        
    Example:
        >>> model, config = load_quantized_mixtral_model(
        ...     "/path/to/quantized_model",
        ...     device_map="auto"
        ... )
        >>> # 现在可以进行推理
        >>> outputs = model.generate(input_ids=...)
    """
    import glog
    import sys
    from lib import codebook, utils
    from model.mixtral_moe import MixtralConfig, MixtralForCausalLM
    
    if model_class is None:
        model_class = MixtralForCausalLM
    
    if verbose:
        glog.info(f'Loading quantized model from {quantized_path}')
    
    # 1. 验证路径和加载配置
    assert os.path.exists(quantized_path), f"Quantized model path not found: {quantized_path}"
    saved_config = torch.load(
        os.path.join(quantized_path, 'config.pt'), 
        weights_only=False
    )
    
    # 2. 创建模型配置
    model_config_dict = saved_config['model_config']
    model_config = MixtralConfig(**model_config_dict)
    
    if verbose:
        glog.info(f'Model config loaded: {model_config.model_type}')
    
    # 3. 提取量化参数
    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']
    codebook_version = model_config.quip_params.get(
        'codebook_version', 
        model_config.quip_params['codebook']
    )
    num_experts = model_config.num_local_experts
    
    if verbose:
        glog.info(f'Quantization params - Codebook: {codebook_id}, Codesz: {codesz}')
        glog.info(f'Model has {num_experts} experts per layer')
    
    # 4. 初始化模型（这会创建所有QuantizedLinear层）
    model = model_class.from_pretrained(
        model_config._name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        config=model_config
    ).half()
    
    if verbose:
        glog.info('Model initialized with QuantizedLinear layers')
    
    cpu = torch.device('cpu')
    
    # 5. 加载非量化的权重（embeddings, norms, lm_head等）
    
    # 加载lm_head
    if os.path.exists(f'{quantized_path}/lm_head.pt'):
        lmhead_data = torch.load(
            f'{quantized_path}/lm_head.pt',
            map_location=cpu, 
            weights_only=False
        )
        model.lm_head.weight.copy_(
            lmhead_data['weight'].to(model.lm_head.weight.device)
        )
        if verbose:
            glog.info('✓ Loaded lm_head')
    
    # 加载final norm
    if os.path.exists(f'{quantized_path}/final_norm.pt'):
        norm_data = torch.load(
            f'{quantized_path}/final_norm.pt',
            map_location=cpu, 
            weights_only=False
        )
        model.model.norm.weight.copy_(
            norm_data['weight'].to(model.model.norm.weight.device)
        )
        if verbose:
            glog.info('✓ Loaded final_norm')
    
    # 加载embeddings
    if os.path.exists(f'{quantized_path}/embed.pt'):
        embed_data = torch.load(
            f'{quantized_path}/embed.pt',
            map_location=cpu, 
            weights_only=False
        )
        model.model.embed_tokens.weight.copy_(
            embed_data['weight'].to(model.model.embed_tokens.weight.device)
        )
        if verbose:
            glog.info('✓ Loaded embeddings')
    
    # 6. 逐层加载量化权重
    num_layers = len(model.model.layers)
    
    for layer_idx in range(num_layers):
        if verbose:
            print(f'\n{"="*60}')
            print(f'Processing layer {layer_idx}/{num_layers-1}')
            print(f'{"="*60}')
        
        layer = model.model.layers[layer_idx]
        
        # 加载LayerNorm权重
        if os.path.exists(f'{quantized_path}/{layer_idx}_layernorm.pt'):
            ln_data = torch.load(
                f'{quantized_path}/{layer_idx}_layernorm.pt',
                map_location=cpu, 
                weights_only=False
            )
            layer.input_layernorm.weight.copy_(
                ln_data['input_layernorm'].to(layer.input_layernorm.weight.device)
            )
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(layer.post_attention_layernorm.weight.device)
            )
            if verbose:
                glog.info(f'✓ Loaded LayerNorms for layer {layer_idx}')
        
        # 加载注意力层的Q/K/V/O投影（量化）
        attn_proj_names = ['q', 'k', 'v', 'o']
        for proj_name in attn_proj_names:
            proj_file = f'{quantized_path}/{layer_idx}_{proj_name}.pt'
            if os.path.exists(proj_file):
                saved_layer = torch.load(proj_file, map_location=cpu, weights_only=False)
                attn_module = getattr(layer.self_attn, f'{proj_name}_proj')
                utils.unpack_quip(
                    attn_module, 
                    saved_layer, 
                    codebook_id, 
                    codesz, 
                    codebook_version
                )
                if verbose:
                    glog.info(f'  ✓ Loaded {proj_name.upper()} projection')
            else:
                if verbose:
                    glog.warning(f'  ✗ {proj_name.upper()} projection file not found')
        
        # 加载gate层（FP16）
        if os.path.exists(f'{quantized_path}/{layer_idx}_gate.pt'):
            gate_data = torch.load(
                f'{quantized_path}/{layer_idx}_gate.pt',
                map_location=cpu, 
                weights_only=False
            )
            layer.block_sparse_moe.gate.weight.copy_(
                gate_data['weight'].to(layer.block_sparse_moe.gate.weight.device)
            )
            if verbose:
                glog.info(f'✓ Loaded gate for layer {layer_idx}')
        
        # 加载专家层（量化）
        for expert_idx in range(num_experts):
            expert = layer.block_sparse_moe.experts[expert_idx]
            
            for expert_proj in ['w1', 'w2', 'w3']:
                expert_file = f'{quantized_path}/{layer_idx}_expert{expert_idx}_{expert_proj}.pt'
                if os.path.exists(expert_file):
                    saved_layer = torch.load(expert_file, map_location=cpu, weights_only=False)
                    expert_module = getattr(expert, expert_proj)
                    utils.unpack_quip(
                        expert_module,
                        saved_layer,
                        codebook_id,
                        codesz,
                        codebook_version
                    )
                    if verbose:
                        glog.info(f'  ✓ Loaded {expert_proj} for expert {expert_idx}')
                else:
                    if verbose:
                        glog.warning(f'  ✗ {expert_proj} file not found for expert {expert_idx}')
        
        if verbose:
            glog.info(f'✓ Completed layer {layer_idx}')
    
    if verbose:
        glog.info('\n' + '='*60)
        glog.info('✓ All layers loaded successfully!')
        glog.info('='*60)
    
    model.eval()  # 设置为评估模式
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    return model, model_config


def load_quantized_mixtral_hf_model(
    quantized_path,
    device_map="auto",
    torch_dtype=torch.float16,
    model_class=None,
    verbose=True
):
    from lib.utils.unsafe_import import model_from_hf_path
    model, _ = model_from_hf_path(quantized_path, use_cuda_graph=False, use_flash_attn=False)
    if verbose:
        print(model)
    return model


class MixtralE8PLoRAConverter:
    """Mixtral 8x7B E8P量化模型到LoRA模型的转换器"""

    def __init__(
      self, 
      quantized_model_path,
      original_model_path,
      lora_rank=16,
      num_layers=12,
      device='cuda'
    ):
        """
        Args:
            quantized_model_path: E8P量化后的模型路径
            original_model_path: 原始FP16模型路径（用于计算误差）
            lora_rank: LoRA的秩
            num_layers: 转换的层数（默认前12层）
            device: 设备
        """
        self.quantized_model_path = quantized_model_path
        self.original_model_path = original_model_path
        self.lora_rank = lora_rank
        self.num_layers = num_layers
        self.device = device
        
        print(f"初始化转换器:")
        print(f"  - 量化模型: {quantized_model_path}")
        print(f"  - 原始模型: {original_model_path}")
        print(f"  - LoRA秩: {lora_rank}")
        print(f"  - 转换层数: {num_layers}")
    
    def load_tokenizer(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.quantized_model_path)

    def load_models(self):
        """Quantized model and original model loading"""
        print("\n[1/4] 加载模型...")
        
        # 加载量化模型
        print("加载量化模型...")
        # self.quantized_model = torch.load(
            # # self.quantized_model_path,
            # map_location=self.device
        # )
        self.quantized_model = load_quantized_mixtral_hf_model(
            self.quantized_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            verbose=False
        )
        
        # 加载原始模型（仅用于提取权重计算误差）
        print("加载原始FP16模型...")
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.original_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print("✓ 模型加载完成")
        
    def identify_quantized_layers(
        self
    ):
        """identify quantized layers in the model"""
        print("\n[2/4] 识别量化层...")
        
        self.layers_to_convert = []
        
        # Mixtral的层结构
        # model.layers[i].block_sparse_moe.experts[j].{w1, w2, w3}
        # model.layers[i].self_attn.{q_proj, k_proj, v_proj, o_proj}
        
        for layer_idx in range(self.num_layers):
            layer_name = f"model.layers.{layer_idx}"
            
            # 1. MoE专家层
            for expert_idx in range(8):  # Mixtral有8个专家
                for weight_name in ['w1', 'w2', 'w3']:
                    quant_path = f"{layer_name}.block_sparse_moe.experts.{expert_idx}.{weight_name}"
                    orig_path = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{weight_name}"
                    
                    self.layers_to_convert.append({
                        'layer_idx': layer_idx,
                        'expert_idx': expert_idx,
                        'weight_name': weight_name,
                        'quant_path': quant_path,
                        'orig_path': orig_path,
                        'type': 'moe'
                    })
            
            # 2. 注意力层
            for attn_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                quant_path = f"{layer_name}.self_attn.{attn_name}"
                orig_path = f"model.layers.{layer_idx}.self_attn.{attn_name}"
                
                self.layers_to_convert.append({
                    'layer_idx': layer_idx,
                    'attn_name': attn_name,
                    'quant_path': quant_path,
                    'orig_path': orig_path,
                    'type': 'attn'
                })
        
        # print(f"✓ 识别到 {len(self.layers_to_convert)} 个量化层")
        quantized_layers_num = len(self.layers_to_convert) 
        # print(f"  - {self.num_layers * 8 * 3} 个MoE专家层")
        moe_expert_layers_num = self.num_layers * 8 * 3 
        # print(f"  - {self.num_layers * 4} 个注意力层")
        attn_layers_num = self.num_layers * 4
        assert quantized_layers_num == moe_expert_layers_num + attn_layers_num, "识别的层数不匹配预期数量"

        return self.layers_to_convert
    
    def get_module_by_path(self, model, path):
        """根据路径获取模块"""
        parts = path.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def set_module_by_path(self, model, path, new_module):
        """根据路径设置模块"""
        parts = path.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)
    
    def extract_quantization_params(self, quantized_module):
        """从量化模块中提取E8P量化参数
        
        注意：不同的量化模块可能采用不同的存储方式：
        1. 已forward过：Qidxs_list_0, Qidxs_list_1, ... 等多个索引缓冲区
        2. 未forward过：Qidxs（单个原始索引缓冲区），需要解压
        """
        from lib import codebook
        from lib.codebook.latticee8_padded12 import QuantizedE8P12Linear
        
        params = {}
        
        # 1. 提取量化索引
        Qidxs_list = None
        
        # 首先尝试获取已解压的列表
        if hasattr(quantized_module, 'Qidxs_list') and quantized_module.Qidxs_list is not None:
            Qidxs_list = quantized_module.Qidxs_list
            print(f"    ℹ 使用已缓存的 Qidxs_list")
        
        # 尝试从Qidxs_0, Qidxs_1...收集
        if Qidxs_list is None:
            collected_qidxs = []
            i = 0
            while hasattr(quantized_module, f'Qidxs_{i}'):
                collected_qidxs.append(getattr(quantized_module, f'Qidxs_{i}'))
                i += 1
            if collected_qidxs:
                Qidxs_list = collected_qidxs
                print(f"    ℹ 从 Qidxs_0...Qidxs_{i-1} 收集到 {i} 个索引缓冲区")
        
        # 如果还是没有，尝试从原始的Qidxs进行解压
        if Qidxs_list is None and hasattr(quantized_module, 'Qidxs'):
            print(f"    ℹ 从原始 Qidxs 进行解压...")
            try:
                # 获取codebook_id
                codebook_id = quantized_module.codebook_id.item() if hasattr(quantized_module, 'codebook_id') else 17
                
                # 获取模块所在的设备
                device = next(quantized_module.parameters()).device if list(quantized_module.parameters()) else 'cuda'
                
                # 创建 QuantizedE8P12Linear 实例以调用 maybe_unpack_idxs
                quantized_class_instance = QuantizedE8P12Linear(device)
                
                # 调用解压方法
                split_qidxs = quantized_class_instance.maybe_unpack_idxs(quantized_module.Qidxs)
                Qidxs_list = split_qidxs
                print(f"    ✓ 成功解压 Qidxs，得到 {len(split_qidxs)} 个索引缓冲区")
            except Exception as e:
                print(f"    ✗ 解压失败: {e}")
                import traceback
                traceback.print_exc()
                Qidxs_list = None
        
        if Qidxs_list is None:
            raise AttributeError("无法获取任何形式的 Qidxs 索引")
        
        params['Qidxs_list'] = Qidxs_list
        
        # 2. 提取缩放参数（必须）
        for attr_name in ['SU', 'SV']:
            if hasattr(quantized_module, attr_name):
                params[attr_name] = getattr(quantized_module, attr_name)
            else:
                raise AttributeError(f"找不到缩放参数 {attr_name}")
        
        # 3. 提取Hadamard相关参数
        for attr_name in ['had_left_T', 'had_left', 'had_right', 'K_left', 'K_right']:
            if hasattr(quantized_module, attr_name):
                val = getattr(quantized_module, attr_name)
                # 映射had_left_T到had_left（如果需要转置）
                if attr_name == 'had_left_T' and val is not None:
                    params['had_left'] = val.T.contiguous() if hasattr(val, 'T') else val
                elif attr_name != 'had_left_T':
                    params[attr_name] = val
            else:
                if attr_name == 'had_left_T':
                    params['had_left'] = None
                else:
                    params[attr_name] = None
        
        # 4. 提取bias（可选）
        if hasattr(quantized_module, 'bias') and quantized_module.bias is not None:
            params['bias'] = quantized_module.bias
        else:
            params['bias'] = None
        
        return params
    
    def convert_single_layer(self, layer_info):
        """ convert a single quantized layer to a LoRA layer using the extracted parameters
        
        Args:
            layer_info: 包含层信息的字典（quant_path, orig_path等）
            
        Returns:
            转换后的LoRA模块，或None如果转换失败
        """
        quant_path = layer_info['quant_path']
        orig_path = layer_info['orig_path']
        
        # 1. 获取量化模块和原始模块
        quantized_module = self.get_module_by_path(self.quantized_model, quant_path)
        original_module = self.get_module_by_path(self.original_model, orig_path)
        
        # 2. 验证量化模块类型（从HF模型加载时，应该是某种量化Linear）
        # 注意：可能不是QuantizedLinear而是其他有量化参数的模块
        if not isinstance(quantized_module, QuantizedLinear):
            print(f"⚠ 跳过非E8P层: {quant_path}, 实际类型: {type(quantized_module).__name__}")
            return None
        
        # 3. 获取原始权重
        if not hasattr(original_module, 'weight'):
            print(f"⚠ 无法获取原始权重: {orig_path}")
            return None
            
        original_weight = original_module.weight.data.cpu()  # 复制到CPU避免显存溢出
        
        # 4. 提取量化参数（从QuantizedLinear模块）

        quant_params = self.extract_quantization_params(quantized_module)
        # print("quant_params", quant_params.keys())
        # print("quant_params SU shape", quant_params['SU'].shape)
        # print("quant_params SV shape", quant_params['SV'].shape)
        
        # qidxs_list = quant_params['Qidxs_list']
        # print(f"quant_params Qidxs_list shape: {qidxs_list[0].shape}")
        
        # print("quant_params had_left shape", quant_params['had_left'] if quant_params['had_left'] is not None else None)
        # print("quant_params had_right shape", quant_params['had_right']if quant_params['had_right'] is not None else None) 
        # print("K_left", quant_params['K_left']if quant_params['K_left'] is not None else None)
        # print("K_right shape", quant_params['K_right'] if quant_params['K_right'] is not None else None)
        # print("quant_params bias shape", quant_params['bias'].shape if quant_params['bias'] is not None else None)
        # exit() 
        
        
        # 5. 使用提取的参数创建带LoRA的层
        lora_module = add_lora_to_e8p_layer(
            original_module=quantized_module,
            original_weight=original_weight,
            rank=self.lora_rank,
            **quant_params
        )
        
        print(f"✓ 成功转换 {quant_path}")
        return lora_module
            
    
    def convert_all_layers(self):
        """转换所有识别的层"""
        print("\n[3/4] 转换量化层为LoRA层...")
        
        conversion_stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        with tqdm(total=len(self.layers_to_convert), desc="转换进度") as pbar:
            for layer_info in self.layers_to_convert:
                quant_path = layer_info['quant_path']
                
                # 转换单层
                lora_module = self.convert_single_layer(layer_info)
                
                if lora_module is not None:
                    # 替换原有模块
                    self.set_module_by_path(
                        self.quantized_model, 
                        quant_path, 
                        lora_module
                    )
                    conversion_stats['success'] += 1
                    pbar.set_postfix({'成功': conversion_stats['success']})
                else:
                    conversion_stats['failed'] += 1
                
                pbar.update(1)
        
        print(f"\n✓ 转换完成:")
        print(f"  - 成功: {conversion_stats['success']}")
        print(f"  - 失败: {conversion_stats['failed']}")
        print(f"  - 跳过: {conversion_stats['skipped']}")
        
        return conversion_stats
    
    def save_converted_model(self, output_path):
        """保存转换后的模型"""
        print(f"\n[4/4] 保存转换后的模型到 {output_path}...")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型 state_dict（避免序列化问题）
        model_save_path = output_dir / "pytorch_model.pt"
        print(f"正在保存模型 state_dict...")
        torch.save(self.quantized_model.state_dict(), model_save_path)
        
        # 保存配置
        config = {
            'lora_rank': self.lora_rank,
            'num_layers_converted': self.num_layers,
            'num_layers_total': len(self.layers_to_convert),
            'model_type': 'mixtral-8x7b-e8p-lora',
            'original_model_path': self.original_model_path if hasattr(self, 'original_model_path') else 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        }
        
        config_save_path = output_dir / "lora_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ 模型已保存:")
        print(f"  - 模型文件: {model_save_path}")
        print(f"  - 配置文件: {config_save_path}")
    
    def save_converted_hf_model(
        self,
        output_path,
    ):
        """保存转换后的模型为HuggingFace格式（如果需要）"""
        print(f"\n[4/4] 保存转换后的模型到 {output_path} (HuggingFace格式)...")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查并确保所有模块都有 Qidxs（而不是 Qidxs_0）
        print("  检查 Qidxs 参数...")
        qidxs_count = 0
        unpacked_count = 0
        for name, module in self.quantized_model.named_modules():
            if hasattr(module, 'Qidxs'):
                qidxs_count += 1
            if hasattr(module, 'Qidxs_0'):
                unpacked_count += 1
                print(f"    ⚠ 发现 unpacked Qidxs 在 {name}")
        
        print(f"  找到 {qidxs_count} 个 Qidxs 参数（打包版本）")
        if unpacked_count > 0:
            print(f"  ⚠ 警告: 找到 {unpacked_count} 个 unpacked Qidxs (Qidxs_0)")
        
        # 更新模型 config 中的 rank 参数
        if hasattr(self.quantized_model, 'config'):
            print(f"  更新 config 中的 rank 为: {self.lora_rank}")
            self.quantized_model.config.lora_rank = self.lora_rank
            self.quantized_model.config.lora_alpha = 16.0  # 默认值
            # 保存额外的 LoRA 配置信息
            if not hasattr(self.quantized_model.config, 'lora_config'):
                self.quantized_model.config.lora_config = {
                    'rank': self.lora_rank,
                    'alpha': 16.0,
                    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'w1', 'w2', 'w3']
                }
        
        # 直接使用HF的save_pretrained方法保存整个模型
        # safe_serialization=True 确保使用 safetensors 格式并正确设置 metadata
        self.quantized_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        print(f"✓ 模型已保存为 HuggingFace 格式: {output_dir}")
    
    def run(self, output_path):
        """运行完整的转换流程"""
        print("=" * 60)
        print("Mixtral 8x7B E8P -> LoRA 转换器")
        print("=" * 60)
        
        # 1. 加载模型
        self.load_models()
        self.load_tokenizer()
        
        # 2. 识别层
        self.identify_quantized_layers()
        
        # 3. 转换层
        stats = self.convert_all_layers()
        
        # 4. 保存模型
        if stats['success'] > 0:
            self.save_converted_hf_model(output_path) 
        else:
            print("\n✗ 没有成功转换任何层，不保存模型")
            return False
        
        print("\n" + "=" * 60)
        print("转换完成！")
        print("=" * 60)
        
        # 清理内存
        # Test generation
        input_texts = "The capital of France is "
        inputs = self.tokenizer(input_texts, return_tensors="pt").to(self.device)
        with torch.no_grad(): 
            outputs = self.quantized_model.generate(**inputs, max_new_tokens=20)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True) 
        print(f"\n测试生成: {generated_text}") 
        del self.original_model
        torch.cuda.empty_cache()
        
        return True


def load_converted_model(model_dir, device='cuda'):
    """加载转换后的模型"""
    model_dir = Path(model_dir)
    
    # 加载原始模型结构
    print(f"从 {model_dir} 加载模型...")
    config_path = model_dir / "lora_config.json"
    state_dict_path = model_dir / "pytorch_model.pt"
    
    if not config_path.exists() or not state_dict_path.exists():
        raise FileNotFoundError(f"配置或模型文件不存在: {model_dir}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 加载原始模型来获得基础结构
    original_model_path = config.get('original_model_path', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
    print(f"加载原始模型结构: {original_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # 加载转换后的 state_dict
    print(f"加载转换后的权重...")
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    return model


def test_converted_model(model_dir, tokenizer_path):
    """测试转换后的模型"""
    print("\n" + "=" * 60)
    print("测试转换后的模型")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    model = load_converted_model(model_dir, device='cuda')
    model.eval()
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 测试输入
    test_text = "Hello, how are you?"
    print(f"\n测试输入: {test_text}")
    
    inputs = tokenizer(test_text, return_tensors="pt").to('cuda')
    
    # 前向传播
    print("执行前向传播...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✓ 输出形状: {outputs.logits.shape}")
    print(f"✓ 输出范围: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")
    
    # 生成测试
    print("\n生成测试...")
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"生成文本: {generated_text}")
    
    print("\n✓ 模型测试通过")


def main():
    parser = argparse.ArgumentParser(
        description="将Mixtral 8x7B E8P量化模型转换为带LoRA的模型"
    )
    
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="E8P量化后的模型路径"
    )
    
    parser.add_argument(
        "--original_model",
        type=str,
        required=True,
        help="原始FP16模型路径（HuggingFace格式）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA的秩 (默认: 16)"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="转换的层数 (默认: 12)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (默认: cuda)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="转换后测试模型"
    )
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = MixtralE8PLoRAConverter(
        quantized_model_path=args.quantized_model,
        original_model_path=args.original_model,
        lora_rank=args.lora_rank,
        num_layers=args.num_layers,
        device=args.device
    )
    
    # 运行转换
    success = converter.run(args.output_dir)
    
    # 测试（如果需要）
    if success and args.test:
        model_path = os.path.join(args.output_dir, "pytorch_model.bin")
        test_converted_model(model_path, args.original_model)


if __name__ == "__main__":
    main()