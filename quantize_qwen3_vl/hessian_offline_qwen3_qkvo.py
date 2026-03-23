import argparse
import datetime
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import AutoModelForImageTextToText, AutoProcessor

from lib import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def forward_layer_qwen3vl(
    layer,
    position_ids,
    attention_mask,
    bs,
    device,
    dev_emb,
    rope_deltas=None
):
    """Forward pass for Qwen3-VL (MoE) layer - collecting Projections and MoE Hessians."""
    layer = layer.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 1. Register hooks for Attention projections
    print("Collecting Hessian for self_attn: q_proj, k_proj, v_proj, o_proj")
    hooks = {
        'q': utils.register_H_hook(layer.self_attn.q_proj, device),
        'k': utils.register_H_hook(layer.self_attn.k_proj, device),
        'v': utils.register_H_hook(layer.self_attn.v_proj, device),
        'o': utils.register_H_hook(layer.self_attn.o_proj, device),
    }
    
    # 2. Register hooks for MoE layers (Qwen3-VL A3B architecture)
    print("Collecting Hessian for MoE: gate, shared_expert, and routed experts")
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
        # 路由门控网络
        hooks['moe_gate'] = utils.register_H_hook(layer.mlp.gate, device)
        
        # 共享专家 (Shared Expert)
        if hasattr(layer.mlp, 'shared_expert'):
            hooks['shared_gate'] = utils.register_H_hook(layer.mlp.shared_expert.gate_proj, device)
            hooks['shared_up'] = utils.register_H_hook(layer.mlp.shared_expert.up_proj, device)
            hooks['shared_down'] = utils.register_H_hook(layer.mlp.shared_expert.down_proj, device)
            
        # 路由专家 (Routed Experts) - 为了节省显存，这里仅以第一个专家为例，实际量化中可能需要遍历所有专家
        # 如果显存足够，可以循环注册所有 layer.mlp.experts[i]
        if hasattr(layer.mlp, 'experts') and len(layer.mlp.experts) > 0:
            hooks['expert0_gate'] = utils.register_H_hook(layer.mlp.experts[0].gate_proj, device)
            hooks['expert0_up'] = utils.register_H_hook(layer.mlp.experts[0].up_proj, device)
            hooks['expert0_down'] = utils.register_H_hook(layer.mlp.experts[0].down_proj, device)
    else:
        # Fallback for dense layers if any
        hooks['mlp_gate'] = utils.register_H_hook(layer.mlp.gate_proj, device)
        hooks['mlp_up'] = utils.register_H_hook(layer.mlp.up_proj, device)
        hooks['mlp_down'] = utils.register_H_hook(layer.mlp.down_proj, device)

    # Forward pass through layer
    assert len(dev_emb) % bs == 0
    for i in tqdm(range(len(dev_emb) // bs), desc="Forward pass"):
        batch = dev_emb[i * bs:(i + 1) * bs].to(device)
        batch_pos_ids = position_ids[i * bs:(i + 1) * bs].to(device) if position_ids is not None else None
        batch_attn_mask = attention_mask[i * bs:(i + 1) * bs].to(device) if attention_mask is not None else None
        batch_rope_deltas = rope_deltas[i * bs:(i + 1) * bs].to(device) if rope_deltas is not None else None
        
        with torch.no_grad():
            # Qwen3-VL forward arguments
            kwargs = {
                "position_ids": batch_pos_ids,
                "attention_mask": batch_attn_mask,
                "use_cache": False,
                "output_attentions": False
            }
            if batch_rope_deltas is not None:
                kwargs["position_embeddings"] = batch_rope_deltas
                
            output = layer(batch, **kwargs)[0]
            
        dev_emb[i * bs:(i + 1) * bs] = output.cpu()

    # Collect results
    layer = layer.cpu()
    results = {key: hook() for key, hook in hooks.items()}

    return results


def save_hessians(results, args, transformer_layer_index):
    """Process and save Hessians for a single layer."""
    for key, (H, mu, ct) in results.items():
        mu = mu / ct
        H = H / ct
        H.sub_(mu.unsqueeze(-1) @ mu.unsqueeze(0))
        
        # Symmetrize
        H = (H + H.T) / 2
        
        # Ensure positive semi-definiteness
        try:
            eigvals = torch.linalg.eigvalsh(H)
            min_eigval = eigvals.min().item()
            if min_eigval < 0:
                H.add_(torch.eye(H.shape[0], dtype=H.dtype, device=H.device) * abs(min_eigval) * 1.1)
        except:
            pass

        save_path = f"{args.save_path}/{transformer_layer_index}_{key}.pt"

        torch.save({
            'flatH': utils.sym_to_flat(H.to(torch.float32)),
            'mu': mu.to(torch.float32),
            'n': H.shape[0],
            'ct': ct
        }, save_path)

        print(f"Saved {save_path}")

    del results


def prepare_sample_inputs(processor, num_samples):
    """Prepare sample text-only inputs for Qwen3-VL."""
    messages_list = []
    for _ in range(num_samples):
        messages_list.append([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the importance of artificial intelligence in modern society."}
                ]
            }
        ])
    
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
             for msg in messages_list]
    
    inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt"
    )
    return inputs


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    # 使用 AutoModelForCausalLM 以兼容最新的 Qwen3-VL 架构
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("Loaded model!")

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    # Load or create dataset
    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("Loading cached dataset...")
        loaded_dev_activations = torch.load(f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        position_ids = loaded_dev_activations.get('position_ids', None)
        attention_mask = loaded_dev_activations.get('attention_mask', None)
        rope_deltas = loaded_dev_activations.get('rope_deltas', None)
        print(f"Loaded cached dataset from {loaded_dev_activations['timestamp']}")
    else:
        print("Preparing sample inputs...")
        inputs = prepare_sample_inputs(processor, args.devset_size)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1)
        
        with torch.no_grad():
            dev_emb = model.model.embed_tokens(input_ids.to(device)).cpu()
        
        # Qwen3-VL 可能会使用复杂的 3D RoPE，如果纯文本校准，通常 position_ids 即可
        rope_deltas = None 
        
        after_layer = -1
        print("Prepared dataset!")

    print(f"dev_emb shape: {dev_emb.shape}, dtype: {dev_emb.dtype}")

    num_layers = len(model.model.layers)
    for transformer_layer_index in range(num_layers):
        if transformer_layer_index <= after_layer:
            print(f"Skipping layer {transformer_layer_index}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing layer {transformer_layer_index}/{num_layers - 1}")
        print(f"{'='*50}")

        transformer_layer = model.model.layers[transformer_layer_index]

        results = forward_layer_qwen3vl(
            transformer_layer,
            position_ids,
            attention_mask,
            args.batch_size,
            device,
            dev_emb,
            rope_deltas
        )

        save_hessians(results, args, transformer_layer_index)

        transformer_layer.cpu()
        model.model.layers[transformer_layer_index] = None
        utils.clean()

        if args.save_activations and (
            transformer_layer_index % args.act_save_rate == 0 or
            transformer_layer_index == num_layers - 1
        ):
            torch.save({
                'dev_emb': dev_emb,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'rope_deltas': rope_deltas,
                'after_layer': transformer_layer_index,
                'timestamp': str(datetime.datetime.now())
            }, f'{args.save_path}/dev_activations.pt')
            print(f"Saved activation checkpoint at layer {transformer_layer_index}")

    print("\n" + "="*50)
    print("All layers processed successfully!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--devset_size', default=256, type=int)
    parser.add_argument('--base_model', default='Qwen/Qwen3-VL-30B-A3B-Instruct', type=str)
    parser.add_argument('--save_path', default='hessians/qwen3_vl_30b_a3b', type=str)
    parser.add_argument('--act_save_rate', default=4, type=int)
    parser.add_argument('--save_activations', action='store_true')

    torch.set_grad_enabled(False)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)
    main(args)