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


def forward_layer_mixtral(
    layer,
    position_ids,
    attention_mask,
    bs,
    device,
    dev_emb,
    num_experts
):
    """Forward pass for Mixtral layer - collecting Q/K/V/O projection Hessians."""
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Register hooks for Q, K, V, O projections
    print("Collecting Hessian for self_attn: q_proj, k_proj, v_proj, o_proj")
    done_q = utils.register_H_hook(layer.self_attn.q_proj, device)
    done_k = utils.register_H_hook(layer.self_attn.k_proj, device)
    done_v = utils.register_H_hook(layer.self_attn.v_proj, device)
    done_o = utils.register_H_hook(layer.self_attn.o_proj, device)
    
    # Expert layers (commented out - not collecting yet)
    # done_gate = utils.register_H_hook(layer.block_sparse_moe.gate, device)
    # 
    # done_experts = {}
    # for expert_idx in range(num_experts):
    #     expert = layer.block_sparse_moe.experts[expert_idx]
    #     done_experts[f'expert{expert_idx}_w1'] = utils.register_H_hook(expert.w1, device)
    #     done_experts[f'expert{expert_idx}_w2'] = utils.register_H_hook(expert.w2, device)
    #     done_experts[f'expert{expert_idx}_w3'] = utils.register_H_hook(expert.w3, device)

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
                output_attentions=False,
                output_router_logits=False
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
    
    # Expert layers (commented out - not collecting yet)
    # results['gate'] = done_gate()
    # for key, done_fn in done_experts.items():
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
        # This handles numerical precision issues from floating-point arithmetic
        H = (H + H.T) / 2
        
        # Clamp small negative eigenvalues to zero (numerical precision artifacts)
        # These would be negligible in magnitude anyway
        try:
            eigvals = torch.linalg.eigvalsh(H)
            min_eigval = eigvals.min().item()
            if min_eigval < 0:
                # Add small regularization to ensure positive semi-definiteness
                H.add_(torch.eye(H.shape[0], dtype=H.dtype, device=H.device) * abs(min_eigval) * 1.1)
        except:
            # If eigenvalue computation fails, skip this step
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


def count_linear_layers_mixtral(layer, num_experts):
    """Count expected linear layers in Mixtral - collecting Q/K/V/O projections."""
    # Counting: q_proj, k_proj, v_proj, o_proj = 4 layers
    # Not collecting: gate + experts*(w1,w2,w3)
    return 4  # q, k, v, o


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
    print("Loaded model!")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Detect number of experts from model config
    if hasattr(model.config, 'num_local_experts'):
        args.num_experts = model.config.num_local_experts
        print(f"Detected {args.num_experts} experts from model config")

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

    # Prepare position ids and attention mask
    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :].expand(args.batch_size, -1).contiguous()

    if hasattr(model.config, 'sliding_window') and model.config.sliding_window is not None:
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            dev_emb[0:args.batch_size], 0,
            sliding_window=model.config.sliding_window
        )
    else:
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
        expected_count = count_linear_layers_mixtral(transformer_layer, args.num_experts)
        print(f"Found {linear_count} linear layers, collecting Hessians for {expected_count} layers (Q/K/V/O)")
        
        assert hasattr(transformer_layer, 'block_sparse_moe')
        assert hasattr(transformer_layer.block_sparse_moe, 'experts')
        assert len(transformer_layer.block_sparse_moe.experts) == args.num_experts

        # Forward pass and collect Hessians
        results = forward_layer_mixtral(
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
    parser.add_argument('--base_model', default='mistralai/Mixtral-8x7B-v0.1', type=str)
    parser.add_argument('--save_path', default='hessians/mixtral_8x7b', type=str)
    parser.add_argument('--act_save_rate', default=4, type=int)
    parser.add_argument('--save_activations', action='store_true')
    parser.add_argument('--sample_proc', default=8, type=int)
    parser.add_argument('--num_experts', default=8, type=int)

    torch.set_grad_enabled(False)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)
    main(args)