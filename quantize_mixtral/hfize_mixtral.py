import argparse
import os
import sys
import time

# 添加父目录到Python路径，以便能够导入lib模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.utils.model_version import MODEL_VERSION
from model.mixtral_moe import MixtralConfig, MixtralForCausalLM

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', default="/fact_home/zeyuli/quip_sharp/quantized_mixtral_qkvo", type=str)
parser.add_argument('--hf_output_path', default="./mixtral_8x7b_quip", type=str)


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    # 将字典转换为 MixtralConfig 对象
    model_config_dict = saved_config['model_config']
    model_config = MixtralConfig(**model_config_dict)
    
    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']
    codebook_version = model_config.quip_params.get('codebook_version', model_config.quip_params['codebook'])

    num_experts = model_config.num_local_experts
    glog.info(f'Model has {num_experts} experts per layer')

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model_config.quip_params['model_version'] = MODEL_VERSION
    
    # 使用自定义的 MixtralForCausalLM（已经包含 QuantizedLinear 层）
    from model.mixtral_moe import MixtralForCausalLM
    
    # 使用保存的配置直接创建模型，这样 QuantizedLinear 层会使用正确的参数
    # 添加 device_map="auto" 来自动分配到多个 GPU/CPU，避免 OOM
    model = MixtralForCausalLM.from_pretrained(
        model_config._name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        config=model_config
    ).half()
    glog.info('Model loaded with QuantizedLinear layers and device_map=auto')

    cpu = torch.device('cpu')

    if os.path.exists(f'{args.quantized_path}/lm_head.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lm_head.pt',
                                 map_location=cpu, weights_only=False)
        model.lm_head.weight.copy_(lmhead_data['weight'].to(model.lm_head.weight.device))
        glog.info('loaded lm_head')
    
    if os.path.exists(f'{args.quantized_path}/final_norm.pt'):
        norm_data = torch.load(f'{args.quantized_path}/final_norm.pt',
                               map_location=cpu, weights_only=False)
        model.model.norm.weight.copy_(norm_data['weight'].to(model.model.norm.weight.device))
        glog.info('loaded final norm')
    
    if os.path.exists(f'{args.quantized_path}/embed.pt'):
        embed_data = torch.load(f'{args.quantized_path}/embed.pt',
                                map_location=cpu, weights_only=False)
        model.model.embed_tokens.weight.copy_(embed_data['weight'].to(model.model.embed_tokens.weight.device))
        glog.info('loaded embeddings')

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]
        glog.info(f'='*60)
        glog.info(f'Processing layer {ii}/{len(model.model.layers)-1}')
        glog.info(f'='*60)

        # 加载 LayerNorm 权重
        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu, weights_only=False)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(layer.input_layernorm.weight.device))
            layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'].to(layer.post_attention_layernorm.weight.device))
        

        # 加载注意力层的 Q 投影
        q_file = f'{args.quantized_path}/{ii}_q.pt'
        if os.path.exists(q_file):
            glog.info(f'Loading Q projection from {q_file}')
            saved_layer = torch.load(q_file, map_location=cpu, weights_only=False)
            # 直接 unpack 到已存在的 QuantizedLinear 层
            utils.unpack_quip(layer.self_attn.q_proj, saved_layer, codebook_id, codesz, codebook_version)
            glog.info(f'✓ Loaded Q projection for layer {ii}')
        else:
            glog.warning(f'✗ Q projection file not found: {q_file}')

        # 加载注意力层的 K 投影
        k_file = f'{args.quantized_path}/{ii}_k.pt'
        if os.path.exists(k_file):
            glog.info(f'Loading K projection from {k_file}')
            saved_layer = torch.load(k_file, map_location=cpu, weights_only=False)
            # 直接 unpack 到已存在的 QuantizedLinear 层
            utils.unpack_quip(layer.self_attn.k_proj, saved_layer, codebook_id, codesz, codebook_version)
            glog.info(f'✓ Loaded K projection for layer {ii}')
        else:
            glog.warning(f'✗ K projection file not found: {k_file}')

        # 加载注意力层的 V 投影 
        v_file = f'{args.quantized_path}/{ii}_v.pt'
        if os.path.exists(v_file):
            glog.info(f'Loading V projection from {v_file}')
            saved_layer = torch.load(v_file, map_location=cpu, weights_only=False)
            # 直接 unpack 到已存在的 QuantizedLinear 层
            utils.unpack_quip(layer.self_attn.v_proj, saved_layer, codebook_id, codesz, codebook_version)
            glog.info(f'✓ Loaded V projection for layer {ii}')
        else:
            glog.warning(f'✗ V projection file not found: {v_file}')

        # 加载注意力层的 O 投影
        o_file = f'{args.quantized_path}/{ii}_o.pt'
        if os.path.exists(o_file):
            glog.info(f'Loading O projection from {o_file}')
            saved_layer = torch.load(o_file, map_location=cpu, weights_only=False)
            # 直接 unpack 到已存在的 QuantizedLinear 层
            utils.unpack_quip(layer.self_attn.o_proj, saved_layer, codebook_id, codesz, codebook_version)
            glog.info(f'✓ Loaded O projection for layer {ii}')
        else:
            glog.warning(f'✗ O projection file not found: {o_file}')

        glog.info(f'Completed loading Q/K/V/O quantized projections for layer {ii}')

        # 加载 gate 层（FP16）
        if os.path.exists(f'{args.quantized_path}/{ii}_gate.pt'):
            gate_data = torch.load(f'{args.quantized_path}/{ii}_gate.pt',
                                   map_location=cpu, weights_only=False)
            layer.block_sparse_moe.gate.weight.copy_(gate_data['weight'].to(layer.block_sparse_moe.gate.weight.device))
            glog.info(f'loaded layer {ii} gate (FP16)')
        
        # 加载专家层的量化权重（直接 unpack 到已存在的 QuantizedLinear）
        for expert_idx in range(num_experts):
            expert = layer.block_sparse_moe.experts[expert_idx]

            # 加载 w1 (gate projection)
            w1_path = f'{args.quantized_path}/{ii}_expert{expert_idx}_w1.pt'
            if os.path.exists(w1_path):
                saved_layer = torch.load(w1_path, map_location=cpu, weights_only=False)
                utils.unpack_quip(expert.w1, saved_layer, codebook_id, codesz, codebook_version)
                glog.info(f'✓ Loaded quantized w1 for expert {expert_idx} layer {ii}')
            else:
                glog.warning(f'✗ w1 file not found: {w1_path}')

            # 加载 w2 (down projection)
            w2_path = f'{args.quantized_path}/{ii}_expert{expert_idx}_w2.pt'
            if os.path.exists(w2_path):
                saved_layer = torch.load(w2_path, map_location=cpu, weights_only=False)
                utils.unpack_quip(expert.w2, saved_layer, codebook_id, codesz, codebook_version)
                glog.info(f'✓ Loaded quantized w2 for expert {expert_idx} layer {ii}')
            else:
                glog.warning(f'✗ w2 file not found: {w2_path}')

            # 加载 w3 (up projection)
            w3_path = f'{args.quantized_path}/{ii}_expert{expert_idx}_w3.pt'
            if os.path.exists(w3_path):
                saved_layer = torch.load(w3_path, map_location=cpu, weights_only=False)
                utils.unpack_quip(expert.w3, saved_layer, codebook_id, codesz, codebook_version)
                glog.info(f'✓ Loaded quantized w3 for expert {expert_idx} layer {ii}')
            else:
                glog.warning(f'✗ w3 file not found: {w3_path}')

        glog.info(f'loaded layer {ii} all {num_experts} experts (quantized)')

    glog.info('All layers loaded successfully!')
    
    glog.info('saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    glog.info(f'✓ Model successfully saved to {args.hf_output_path}')
    
    # 清理原始模型以释放内存
    del model
    torch.cuda.empty_cache()

    # 验证加载
    glog.info('Reloading model to verify...')
    from lib.utils.unsafe_import import model_from_hf_path
    model, _ = model_from_hf_path(args.hf_output_path, use_cuda_graph=False, use_flash_attn=False)

    glog.info('Successfully loaded hfized model')
    glog.info('Generating some text...')

    start = time.time()
    prompt = 'The capital of France is'

    inputs = tokenizer(prompt, return_tensors='pt')
    print(model)

    outputs = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        attention_mask=inputs['attention_mask'].cuda(),
        max_new_tokens=64,
        return_dict_in_generate=True
    )

    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    glog.info("The output is: " + output_str)
    glog.info(f'Generation elapsed: {time.time() - start:.2f}s')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)