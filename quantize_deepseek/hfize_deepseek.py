import argparse
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glog
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from lib import codebook, utils
from lib.utils.model_version import MODEL_VERSION
from model.configuration_deepseek import DeepseekConfig
from model.deepseek_moe import DeepseekForCausalLM

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
torch.set_grad_enabled(False)


parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', default='/fact_home/zeyuli/quip_sharp/deepseek_quip_full', type=str)
parser.add_argument('--hf_output_path', default='./deepseek_moe_quip', type=str)
parser.add_argument('--base_model', default=None, type=str,
                    help='Optional override for base model path/name. Defaults to config model path.')
parser.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16', 'float32'], type=str)
parser.add_argument('--device_map', default='auto', type=str)
parser.add_argument('--run_generation_test', action='store_true',
                    help='Run a quick generation sanity test after save (optional).')


def _get_dtype(name: str):
    if name == 'float16':
        return torch.float16
    if name == 'bfloat16':
        return torch.bfloat16
    return torch.float32


def _load_quantized_linear_if_exists(path, target_linear, codebook_id, codesz, codebook_version):
    if not os.path.exists(path):
        return False
    saved_layer = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    utils.unpack_quip(target_linear, saved_layer, codebook_id, codesz, codebook_version)
    return True


def _load_gate_if_exists(path, gate_module):
    if not os.path.exists(path):
        return False
    gate_data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(gate_data, dict):
        # Support both full state_dict and {'weight': ...} formats.
        if 'weight' in gate_data and hasattr(gate_module, 'weight'):
            gate_module.weight.copy_(gate_data['weight'].to(gate_module.weight.device))
        else:
            gate_module.load_state_dict(gate_data, strict=False)
    return True


def main(args):
    if not os.path.exists(args.quantized_path):
        raise FileNotFoundError(f'quantized_path not found: {args.quantized_path}')

    config_path = os.path.join(args.quantized_path, 'config.pt')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.pt not found: {config_path}')

    saved_config = torch.load(config_path, weights_only=False)
    model_config_dict = saved_config['model_config']
    model_config = DeepseekConfig(**model_config_dict)

    if getattr(model_config, 'quip_params', None) is None:
        raise ValueError('model_config.quip_params is missing. This does not look like a quantized checkpoint.')

    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']
    codebook_version = model_config.quip_params.get('codebook_version', model_config.quip_params['codebook'])

    num_experts = getattr(model_config, 'n_routed_experts', 0)
    glog.info(f'Model has {num_experts} routed experts per MoE layer')

    base_model_path = args.base_model or model_config._name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model_config.quip_params['model_version'] = MODEL_VERSION

    dtype = _get_dtype(args.dtype)
    model = DeepseekForCausalLM.from_pretrained(
        base_model_path,
        config=model_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    cpu = torch.device('cpu')

    lm_head_path = f'{args.quantized_path}/lm_head.pt'
    final_norm_path = f'{args.quantized_path}/final_norm.pt'
    embed_path = f'{args.quantized_path}/embed.pt'

    if os.path.exists(lm_head_path):
        lmhead_data = torch.load(lm_head_path, map_location=cpu, weights_only=False)
        model.lm_head.weight.copy_(lmhead_data['weight'].to(model.lm_head.weight.device))
        glog.info('Loaded lm_head')

    if os.path.exists(final_norm_path):
        norm_data = torch.load(final_norm_path, map_location=cpu, weights_only=False)
        model.model.norm.weight.copy_(norm_data['weight'].to(model.model.norm.weight.device))
        glog.info('Loaded final_norm')

    if os.path.exists(embed_path):
        embed_data = torch.load(embed_path, map_location=cpu, weights_only=False)
        model.model.embed_tokens.weight.copy_(embed_data['weight'].to(model.model.embed_tokens.weight.device))
        glog.info('Loaded embeddings')

    total_loaded = 0
    total_missing = 0

    for ii in tqdm(range(len(model.model.layers)), desc='Loading quantized layers'):
        layer = model.model.layers[ii]
        glog.info('=' * 60)
        glog.info(f'Processing layer {ii}/{len(model.model.layers) - 1}')

        ln_path = f'{args.quantized_path}/{ii}_layernorm.pt'
        if os.path.exists(ln_path):
            ln_data = torch.load(ln_path, map_location=cpu, weights_only=False)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(layer.input_layernorm.weight.device))
            layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'].to(layer.post_attention_layernorm.weight.device))

        # Attention Q/K/V/O
        for proj_name in ['q', 'k', 'v', 'o']:
            proj_path = f'{args.quantized_path}/{ii}_{proj_name}.pt'
            target_proj = getattr(layer.self_attn, f'{proj_name}_proj')
            if _load_quantized_linear_if_exists(proj_path, target_proj, codebook_id, codesz, codebook_version):
                total_loaded += 1
            else:
                total_missing += 1

        is_moe_layer = hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts') and layer.mlp.experts is not None
        if is_moe_layer:
            moe_module = layer.mlp

            gate_path = f'{args.quantized_path}/{ii}_gate.pt'
            if _load_gate_if_exists(gate_path, moe_module.gate):
                total_loaded += 1

            if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    path = f'{args.quantized_path}/{ii}_shared_{proj}.pt'
                    target_proj = getattr(moe_module.shared_experts, proj)
                    if _load_quantized_linear_if_exists(path, target_proj, codebook_id, codesz, codebook_version):
                        total_loaded += 1
                    else:
                        total_missing += 1

            for expert_idx in range(len(moe_module.experts)):
                expert = moe_module.experts[expert_idx]
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    path = f'{args.quantized_path}/{ii}_expert{expert_idx}_{proj}.pt'
                    target_proj = getattr(expert, proj)
                    if _load_quantized_linear_if_exists(path, target_proj, codebook_id, codesz, codebook_version):
                        total_loaded += 1
                    else:
                        total_missing += 1
        else:
            dense_module = layer.mlp
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                path = f'{args.quantized_path}/{ii}_dense_{proj}.pt'
                target_proj = getattr(dense_module, proj)
                if _load_quantized_linear_if_exists(path, target_proj, codebook_id, codesz, codebook_version):
                    total_loaded += 1
                else:
                    total_missing += 1

    glog.info(f'Finished loading quantized weights: loaded={total_loaded}, missing={total_missing}')

    os.makedirs(args.hf_output_path, exist_ok=True)
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    glog.info(f'Saved HF model to {args.hf_output_path}')

    if args.run_generation_test:
        glog.info('Running quick generation sanity test...')
        test_device = next(model.parameters()).device
        inputs = tokenizer('The capital of France is', return_tensors='pt')
        inputs = {k: v.to(test_device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=32)
        glog.info('Output: ' + tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)