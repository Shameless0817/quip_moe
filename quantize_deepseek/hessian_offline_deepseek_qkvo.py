import argparse
import datetime
import os
import random
import sys

# Add parent directory to path to allow importing lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from lib import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def forward_layer_deepseek(
    layer,
    position_ids,
    attention_mask,
    bs,
    device,
    dev_emb,
    num_experts
):
    """Forward pass for DeepSeek layer - collecting Q/K/V/O projection Hessians."""
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Register hooks for Q, K, V, O projections
    print("Collecting Hessian for self_attn: q_proj, k_proj, v_proj, o_proj")
    done_q = utils.register_H_hook(layer.self_attn.q_proj, device)
    done_k = utils.register_H_hook(layer.self_attn.k_proj, device)
    done_v = utils.register_H_hook(layer.self_attn.v_proj, device)
    done_o = utils.register_H_hook(layer.self_attn.o_proj, device)
    
    # ---------------------------------------------------------
    # Expert layers (commented out - not collecting yet)
    # DeepSeek MoE structure: mlp.gate, mlp.shared_experts, mlp.experts
    # ---------------------------------------------------------
    # is_moe = hasattr(layer.mlp, 'experts')
    # done_experts_hooks = {}
    # 
    # if is_moe:
    #     done_experts_hooks['gate'] = utils.register_H_hook(layer.mlp.gate, device)
    #     if hasattr(layer.mlp, 'shared_experts') and layer.mlp.shared_experts is not None:
    #         done_experts_hooks['shared_gate_proj'] = utils.register_H_hook(layer.mlp.shared_experts.gate_proj, device)
    #         done_experts_hooks['shared_up_proj'] = utils.register_H_hook(layer.mlp.shared_experts.up_proj, device)
    #         done_experts_hooks['shared_down_proj'] = utils.register_H_hook(layer.mlp.shared_experts.down_proj, device)
    #         
    #     for expert_idx in range(num_experts):
    #         expert = layer.mlp.experts[expert_idx]
    #         done_experts_hooks[f'expert{expert_idx}_gate_proj'] = utils.register_H_hook(expert.gate_proj, device)
    #         done_experts_hooks[f'expert{expert_idx}_up_proj'] = utils.register_H_hook(expert.up_proj, device)
    #         done_experts_hooks[f'expert{expert_idx}_down_proj'] = utils.register_H_hook(expert.down_proj, device)
    # else:
    #     # Dense layer
    #     done_experts_hooks['dense_gate_proj'] = utils.register_H_hook(layer.mlp.gate_proj, device)
    #     done_experts_hooks['dense_up_proj'] = utils.register_H_hook(layer.mlp.up_proj, device)
    #     done_experts_hooks['dense_down_proj'] = utils.register_H_hook(layer.mlp.down_proj, device)
    # ---------------------------------------------------------

    # Forward pass through layer
    assert len(dev_emb) % bs == 0
    for i in tqdm(range(len(dev_emb) // bs), desc="Forward pass"):
        # Create batch with correct device and shape
        batch = dev_emb[i * bs:(i + 1) * bs].to(device)
        batch_pos_ids = position_ids[:batch.shape[0]].to(device)
        batch_attn_mask = attention_mask[:batch.shape[0]] if attention_mask.shape[0] > batch.shape[0] else attention_mask
        
        with torch.no_grad():
            output = layer(
                batch,
                position_ids=batch_pos_ids,
                attention_mask=batch_attn_mask,
                use_cache=False,
                output_attentions=False
            )[0]
        dev_emb[i * bs:(i + 1) * bs] = output.cpu()

    # Collect results - Q/K/V/O projections
    layer = layer.cpu()
    
    results = {
        'q': done_q(),
        'k': done_k(),
        'v': done_v(),
        'o': done_o(),
    }
    
    # Collect expert results (commented out)
    # for key, done_fn in done_experts_hooks.items():
    #     results[key] = done_fn()

    return results


def save_hessians(results, args, transformer_layer_index):
    """Process and save Hessians for a single layer."""
    for key, (H, mu, ct) in results.items():
        # Compute covariance: E[xx^T] - E[x]E[x]^T
        mu = mu / ct
        H = H / ct
        # Subtract outer product: H = H - mu @ mu^T
        H.sub_(mu.unsqueeze(-1) @ mu.unsqueeze(0))
        
        # Symmetrize the matrix to ensure it's perfectly symmetric
        H = (H + H.T) / 2
        
        # Clamp small negative eigenvalues to zero
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


def count_linear_layers_deepseek(layer, num_experts):
    """Count expected linear layers in DeepSeek - collecting Q/K/V/O projections."""
    # Counting: q_proj, k_proj, v_proj, o_proj = 4 layers
    return 4  


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect number of experts from DeepSeek model config
    if hasattr(model.config, 'n_routed_experts'):
        args.num_experts = model.config.n_routed_experts
        print(f"Detected {args.num_experts} routed experts from model config")

    # Load or create dataset
    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("Loading cached dataset...")
        loaded_dev_activations = torch.load(f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(f"Loaded cached dataset from {loaded_dev_activations['timestamp']}")
    else:
        print("Loading dataset...")
        devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                                   nproc=args.sample_proc)
        dev_emb = model.model.embed_tokens(devset)
        after_layer = -1
        print("Loaded dataset!")

    print(f"dev_emb shape: {dev_emb.shape}, dtype: {dev_emb.dtype}")

    # Prepare position ids and attention mask (DeepSeek uses standard causal mask)
    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :].expand(args.batch_size, -1).contiguous()

    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        dev_emb[0:args.batch_size], 0
    )

    # Process each layer
    num_layers = len(model.model.layers)
    for transformer_layer_index in range(num_layers):
        if transformer_layer_index <= after_layer:
            print(f"Skipping layer {transformer_layer_index}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing layer {transformer_layer_index}/{num_layers - 1}")
        print(f"{'='*50}")

        transformer_layer = model.model.layers[transformer_layer_index]

        # Validate layer structure
        linear_count = len([m for m in transformer_layer.modules()
                           if isinstance(m, torch.nn.Linear)])
        expected_count = count_linear_layers_deepseek(transformer_layer, args.num_experts)
        
        is_moe = hasattr(transformer_layer.mlp, 'experts')
        layer_type = "MoE" if is_moe else "Dense"
        print(f"Layer type: {layer_type}. Found {linear_count} linear layers, collecting Hessians for {expected_count} layers (Q/K/V/O)")

        # Forward pass and collect Hessians
        results = forward_layer_deepseek(
            transformer_layer,
            position_ids,
            attention_mask,
            args.batch_size,
            device,
            dev_emb,
            args.num_experts
        )

        # Save Hessians
        save_hessians(results, args, transformer_layer_index)

        # Clean up
        transformer_layer.cpu()
        model.model.layers[transformer_layer_index] = None
        utils.clean()

        # Save activations checkpoint
        if args.save_activations and (
            transformer_layer_index % args.act_save_rate == 0 or
            transformer_layer_index == num_layers - 1
        ):
            torch.save({
                'dev_emb': dev_emb,
                'after_layer': transformer_layer_index,
                'timestamp': str(datetime.datetime.now())
            }, f'{args.save_path}/dev_activations.pt')
            print(f"Saved activation checkpoint at layer {transformer_layer_index}")

        print(f"Done processing layer {transformer_layer_index}")

    print("\n" + "="*50)
    print("All layers processed successfully!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--devset_size', default=256, type=int)
    parser.add_argument('--ctx_size', default=4096, type=int)
    # Changed default model to DeepSeek MoE
    parser.add_argument('--base_model', default='deepseek-ai/deepseek-moe-16b-base', type=str)
    parser.add_argument('--save_path', default='hessians/deepseek_moe_16b', type=str)
    parser.add_argument('--act_save_rate', default=4, type=int)
    parser.add_argument('--save_activations', action='store_true')
    parser.add_argument('--sample_proc', default=8, type=int)
    parser.add_argument('--num_experts', default=64, type=int)

    torch.set_grad_enabled(False)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)
    main(args)