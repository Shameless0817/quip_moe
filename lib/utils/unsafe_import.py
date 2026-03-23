# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import transformers

from model.llama import LlamaForCausalLM
from model.mixtral_moe import MixtralForCausalLM
from model.mixtral_moe_lora import LRMixtralForCausalLM
from model.deepseek_moe import DeepseekForCausalLM

from . import graph_wrapper
import torch
import json
import os

from transformers import Qwen2Config
from model.new_llama import LlamaForCausalLM
from model.qwen3 import Qwen3ForCausalLM
from model.configuration_deepseek import DeepseekConfig
from transformers.utils import CONFIG_NAME
import json 
from huggingface_hub import hf_hub_download

def _load_config_dict(path_or_repo: str):
    if os.path.isdir(path_or_repo):
        cfg_path = os.path.join(path_or_repo, CONFIG_NAME)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json not found under: {path_or_repo}")
        with open(cfg_path, "r") as f:
            return json.load(f)
    else:
        cfg_path = hf_hub_download(path_or_repo, CONFIG_NAME)
        with open(cfg_path, "r") as f:
            return json.load(f)


def _unwrap_quant_config_dict(d: dict):
    """
    Support quantization output config layout:
    {
      "quant_args": ...,
      "model_config": {...},
      "model_type": "deepseek"
    }
    """
    if isinstance(d, dict) and isinstance(d.get('model_config'), dict):
        inner = dict(d['model_config'])
        if 'model_type' not in inner and 'model_type' in d:
            inner['model_type'] = d['model_type']
        is_quantized = 'quip_params' in inner
        return inner, is_quantized

    is_quantized = isinstance(d, dict) and ('quip_params' in d)
    return d, is_quantized

# def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None, empty_model=False):
#     # AutoConfig fails to read name_or_path correctly
#     try:
#         bad_config = transformers.AutoConfig.from_pretrained(path)
#         is_quantized = hasattr(bad_config, 'quip_params')
#     except ValueError as e:
#         if "qwen3" in str(e).lower():
#             d = _load_config_dict(path)
#             bad_config = Qwen2Config(**d)
#             is_quantized = hasattr(bad_config, 'quip_params')
#     model_type = bad_config.model_type
#     if is_quantized or empty_model:
#         if model_type == 'llama':
#             model_str = transformers.LlamaConfig.from_pretrained(
#                 path)._name_or_path
#             model_cls = LlamaForCausalLM
#         elif model_type == 'qwen3':
#             if not empty_model:
#                 model_str = Qwen2Config.from_pretrained(
#                     path)._name_or_path
#             else:
#                 model_str = path
#             model_cls = Qwen3ForCausalLM
#         else:
#             raise Exception
#     else:
#         if model_type == 'llama':
#             model_str = path
#             from model.llama_fp16 import LlamaForCausalLMFP16
#             model_cls = LlamaForCausalLMFP16
#         elif model_type == 'qwen3':
#             model_str = path
#             from model.qwen3_fp16 import Qwen3ForCausalLMFP16
#             model_cls = Qwen3ForCausalLMFP16

#     if empty_model:
#         model = model_cls(bad_config)
#         dtype = torch.float16
#         model = model.to(dtype=dtype)
#         model.to('cuda')
#         model.eval()
#     else:
#         model = model_cls.from_pretrained(path,
#                                         torch_dtype='auto',
#                                         low_cpu_mem_usage=True,
#                                         attn_implementation='sdpa',
#                                         device_map='cuda')

#     return model, model_str


def model_from_hf_path(path,
                       use_cuda_graph=True,
                       use_flash_attn=True,
                       device_map='auto',
                       empty_model=False):

    def maybe_wrap(use_cuda_graph):
        return (lambda x: graph_wrapper.get_graph_wrapper(x)
                ) if use_cuda_graph else (lambda x: x)

    # AutoConfig fails to read name_or_path correctly
    bad_config = None
    is_quantized = False

    try:
        bad_config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
        is_quantized = hasattr(bad_config, 'quip_params')
    except ValueError as e:
        if "qwen3" in str(e).lower():
            d_raw = _load_config_dict(path)
            d, is_quantized = _unwrap_quant_config_dict(d_raw)
            bad_config = Qwen2Config(**d)
        elif "model type `deepseek`" in str(e).lower() or "keyerror: 'deepseek'" in str(e).lower() or "deepseek" in str(e).lower():
            d_raw = _load_config_dict(path)
            d, is_quantized = _unwrap_quant_config_dict(d_raw)
            bad_config = DeepseekConfig(**d)
        else:
            raise
    except Exception as e:
        # Try to handle other exceptions gracefully
        print(f"Warning: Failed to load config with AutoConfig: {e}")
        print(f"Attempting to load config from config.json directly...")
        try:
            d_raw = _load_config_dict(path)
            d, is_quantized = _unwrap_quant_config_dict(d_raw)
            # Try to infer model type from config
            if 'model_type' in d:
                model_type = d['model_type']
                if model_type == 'qwen3':
                    bad_config = Qwen2Config(**d)
                elif model_type == 'deepseek':
                    bad_config = DeepseekConfig(**d)
                else:
                    # Create a generic config
                    bad_config = transformers.AutoConfig.for_model(model_type, **d)
            else:
                raise ValueError(f"Cannot determine model type from config at {path}")
        except Exception as inner_e:
            raise RuntimeError(f"Failed to load model config from {path}: {inner_e}")
    
    if bad_config is None:
        raise RuntimeError(f"Failed to load config from {path}")
    
    model_type = bad_config.model_type
    if is_quantized or empty_model:
        if model_type == 'llama':
            if not empty_model:
                model_str = transformers.LlamaConfig.from_pretrained(
                    path)._name_or_path
            else:
                model_str = path
            model_cls = LlamaForCausalLM
        elif model_type == 'mixtral':
            if not empty_model:
                model_str = transformers.AutoConfig.from_pretrained(
                    path)._name_or_path
            else:
                model_str = path
            model_cls = MixtralForCausalLM
        elif model_type == 'qwen3':
            if not empty_model:
                model_str = Qwen2Config.from_pretrained(
                    path)._name_or_path
            else:
                model_str = path
            model_cls = Qwen3ForCausalLM
        elif model_type == 'deepseek':
            if not empty_model:
                model_str = getattr(bad_config, '_name_or_path', path)
            else:
                model_str = path
            model_cls = DeepseekForCausalLM
        else:
            raise Exception(f'Unsupported model type for quantization: {model_type}')
    else:
        model_cls = transformers.AutoModelForCausalLM
        model_str = path

    if empty_model:
        model = model_cls(bad_config)
        dtype = torch.float16
        model = model.to(dtype=dtype)
        model.to('cuda')
        model.eval()
    else:
        print(f"\n{'='*80}")
        print(f"[模型加载] 开始从路径加载模型: {path}")
        print(f"[模型加载] 模型类: {model_cls.__name__}")
        print(f"{'='*80}\n")
        
        model = model_cls.from_pretrained(
            path,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
            attn_implementation='sdpa',
            device_map=device_map,
            trust_remote_code=True
        )
        
        # 检查是否有 Qidxs 参数
        qidxs_count = 0
        qidxs_samples = []
        
        for name, param in model.named_parameters():
            if 'Qidxs' in name:
                qidxs_count += 1
                if len(qidxs_samples) < 3:
                    qidxs_samples.append(f"{name} (param, shape={param.shape})")
        
        for name, buffer in model.named_buffers():
            if 'Qidxs' in name:
                qidxs_count += 1
                if len(qidxs_samples) < 3:
                    qidxs_samples.append(f"{name} (buffer, shape={buffer.shape})")
        
        # print(f"\n[模型加载] 模型中包含 {qidxs_count} 个 Qidxs 参数/缓冲区")
        
        if qidxs_count == 0:
            print("⚠️  警告: 模型中没有找到任何 Qidxs 参数！")
            print("这可能意味着:")
            print("  1. 保存的模型文件中缺少 Qidxs")
            print("  2. 加载时某些参数被跳过了")
            print("  3. 模型结构与保存时不匹配")
        print(f"{'='*80}\n")

    return model, model_str


def lora_model_from_hf_path(path,
                       use_cuda_graph=True,
                       use_flash_attn=True,
                       device_map='auto',
                       empty_model=False):

    def maybe_wrap(use_cuda_graph):
        return (lambda x: graph_wrapper.get_graph_wrapper(x)
                ) if use_cuda_graph else (lambda x: x)

    # AutoConfig fails to read name_or_path correctly

    try:
        bad_config = transformers.AutoConfig.from_pretrained(path)
        is_quantized = hasattr(bad_config, 'quip_params')
    except ValueError as e:
        if "qwen3" in str(e).lower():
            d = _load_config_dict(path)
            bad_config = Qwen2Config(**d)
            is_quantized = hasattr(bad_config, 'quip_params')
    
    model_type = bad_config.model_type
    if is_quantized or empty_model:
        if model_type == 'llama':
            if not empty_model:
                model_str = transformers.LlamaConfig.from_pretrained(
                    path)._name_or_path
            else:
                model_str = path
            model_cls = LlamaForCausalLM
        elif model_type == 'mixtral':
            if not empty_model:
                model_str = transformers.AutoConfig.from_pretrained(
                    path)._name_or_path
            else:
                model_str = path
            model_cls = LRMixtralForCausalLM
        elif model_type == 'qwen3':
            if not empty_model:
                model_str = Qwen2Config.from_pretrained(
                    path)._name_or_path
            else:
                model_str = path
            model_cls = Qwen3ForCausalLM
        elif model_type == 'deepseek':
            if not empty_model:
                model_str = transformers.AutoConfig.from_pretrained(
                    path, trust_remote_code=True)._name_or_path
            else:
                model_str = path
            # For LoRA version, we still use DeepseekForCausalLM since there's no LRDeepseekForCausalLM yet
            model_cls = DeepseekForCausalLM
        else:
            raise Exception(f'Unsupported model type for quantization: {model_type}')
    else:
        model_cls = transformers.AutoModelForCausalLM
        model_str = path

    if empty_model:
        model = model_cls(bad_config)
        dtype = torch.float16
        model = model.to(dtype=dtype)
        model.to('cuda')
        model.eval()
    else:
        print(f"\n{'='*80}")
        print(f"[LoRA模型加载] 开始从路径加载模型: {path}")
        print(f"[LoRA模型加载] 模型类: {model_cls.__name__}")
        print(f"{'='*80}\n")
        
        model = model_cls.from_pretrained(
            path,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
            attn_implementation='sdpa',
            device_map=device_map)
        
        # 检查是否有 Qidxs 参数
        qidxs_count = 0
        qidxs_samples = []
        
        for name, param in model.named_parameters():
            if 'Qidxs' in name:
                qidxs_count += 1
                if len(qidxs_samples) < 3:
                    qidxs_samples.append(f"{name} (param, shape={param.shape})")
        
        for name, buffer in model.named_buffers():
            if 'Qidxs' in name:
                qidxs_count += 1
                if len(qidxs_samples) < 3:
                    qidxs_samples.append(f"{name} (buffer, shape={buffer.shape})")
        
        print(f"\n[LoRA模型加载] 模型中包含 {qidxs_count} 个 Qidxs 参数/缓冲区")
        if qidxs_samples:
            print(f"[LoRA模型加载] 示例:")
            for sample in qidxs_samples:
                print(f"  - {sample}")
        
        if qidxs_count == 0:
            print("⚠️  警告: 模型中没有找到任何 Qidxs 参数！")
            print("这可能意味着:")
            print("  1. 保存的模型文件中缺少 Qidxs")
            print("  2. 加载时某些参数被跳过了")
            print("  3. 模型结构与保存时不匹配")
        print(f"{'='*80}\n")

    return model, model_str