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


def forward_layer_qwen2moe_experts(
    layer,
    position_ids,
    attention_mask,
    bs,
    device,
    dev_emb,
    num_experts,
    has_shared_expert=True
):
    """Forward pass for Qwen2-MoE layer - collecting Expert Hessians (gate + experts for each expert)."""
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Qwen2-MoE uses 'mlp' instead of 'block_sparse_moe'
    moe_module = layer.mlp
    
    # Register hooks for gate
    print(f"Collecting Hessian for gate and {num_experts} experts")
    done_gate = utils.register_H_hook(moe_module.gate, device)
    
    done_experts = {}
    
    # Register hooks for routed experts
    for expert_idx in range(num_experts):
        expert = moe_module.experts[expert_idx]
        # Qwen2-MoE experts typically have: gate_proj, up_proj, down_proj
        # (similar to standard LLaMA FFN structure)
        if hasattr(expert, 'gate_proj'):
            done_experts[f'expert{expert_idx}_gate_proj'] = utils.register_H_hook(expert.gate_proj, device)
        if hasattr(expert, 'up_proj'):
            done_experts[f'expert{expert_idx}_up_proj'] = utils.register_H_hook(expert.up_proj, device)
        if hasattr(expert, 'down_proj'):
            done_experts[f'expert{expert_idx}_down_proj'] = utils.register_H_hook(expert.down_proj, device)
    
    # Register hooks for shared expert (if exists)
    if has_shared_expert and hasattr(moe_module, 'shared_expert'):
        shared_expert = moe_module.shared_expert
        if hasattr(shared_expert, 'gate_proj'):
            done_experts['shared_expert_gate_proj'] = utils.register_H_hook(shared_expert.gate_proj, device)
        if hasattr(shared_expert, 'up_proj'):
            done_experts['shared_expert_up_proj'] = utils.register_H_hook(shared_expert.up_proj, device)
        if hasattr(shared_expert, 'down_proj'):
            done_experts['shared_expert_down_proj'] = utils.register_H_hook(shared_expert.down_proj, device)
        print(f"Including shared expert layers")

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
        # 使用 .copy_() 避免就地操作错误
        dev_emb[i * bs:(i + 1) * bs].copy_(output.cpu())

    # Collect results - gate + all experts
    layer = layer.cpu()
    
    results = {
        'gate': done_gate(),
    }
    
    for key, done_fn in done_experts.items():
        results[key] = done_fn()

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


def count_linear_layers_qwen2moe_experts(layer, num_experts, has_shared_expert=True):
    """Count expected linear layers in Qwen2-MoE experts."""
    # gate: 1 layer
    # Each expert has gate_proj, up_proj, down_proj: num_experts * 3 layers
    count = 1 + num_experts * 3
    
    # Shared expert (if exists): gate_proj, up_proj, down_proj: 3 layers
    if has_shared_expert and hasattr(layer.mlp, 'shared_expert'):
        count += 3
    
    return count


def detect_qwen2moe_config(model):
    """Detect Qwen2-MoE specific configuration."""
    config = {
        'num_experts': 8,  # default
        'has_shared_expert': False,
        'is_moe_layer': []
    }
    
    # Try to get num_experts from config
    if hasattr(model.config, 'num_experts'):
        config['num_experts'] = model.config.num_experts
    elif hasattr(model.config, 'num_local_experts'):
        config['num_experts'] = model.config.num_local_experts
    
    # Check which layers are MoE layers (Qwen2-MoE might not have MoE in all layers)
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            config['is_moe_layer'].append(True)
            # Check if this layer has shared expert
            if hasattr(layer.mlp, 'shared_expert'):
                config['has_shared_expert'] = True
        else:
            config['is_moe_layer'].append(False)
    
    return config


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect Qwen2-MoE configuration
    qwen_config = detect_qwen2moe_config(model)
    args.num_experts = qwen_config['num_experts']
    has_shared_expert = qwen_config['has_shared_expert']
    
    print(f"Detected {args.num_experts} experts from model config")
    print(f"Has shared expert: {has_shared_expert}")
    print(f"MoE layers: {sum(qwen_config['is_moe_layer'])}/{len(qwen_config['is_moe_layer'])}")

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
        dev_emb = model.model.embed_tokens(devset).detach()  # 添加 .detach() 避免梯度追踪
        after_layer = -1
        print("Loaded dataset!")
    
    # 确保 dev_emb 不需要梯度
    dev_emb.requires_grad_(False)

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

        # Skip non-MoE layers if only processing MoE layers
        if not qwen_config['is_moe_layer'][transformer_layer_index]:
            if args.moe_only:
                print(f"Skipping non-MoE layer {transformer_layer_index}")
                continue
            else:
                print(f"Warning: Layer {transformer_layer_index} is not a MoE layer")
                continue

        print(f"\n{'='*50}")
        print(f"Processing layer {transformer_layer_index}/{num_layers - 1}")
        print(f"{'='*50}")

        transformer_layer = model.model.layers[transformer_layer_index]

        # Validate layer structure
        linear_count = len([m for m in transformer_layer.modules()
                           if isinstance(m, torch.nn.Linear)])
        expected_count = count_linear_layers_qwen2moe_experts(
            transformer_layer, args.num_experts, has_shared_expert
        )
        print(f"Found {linear_count} linear layers, collecting Hessians for {expected_count} layers (gate + experts)")
        
        assert hasattr(transformer_layer, 'mlp')
        assert hasattr(transformer_layer.mlp, 'gate')
        assert hasattr(transformer_layer.mlp, 'experts')
        assert len(transformer_layer.mlp.experts) == args.num_experts

        # Forward pass and collect Hessians
        print("Running forward pass to collect Hessians...")
        results = forward_layer_qwen2moe_experts(
            transformer_layer,
            position_ids,
            attention_mask,
            args.batch_size,
            device,
            dev_emb,
            args.num_experts,
            has_shared_expert
        )

        # Save Hessians
        print("Saving Hessians...")
        save_hessians(results, args, transformer_layer_index)

        # Save activations if requested
        if args.save_activations:
            print("Saving activations...")
            torch.save({
                'dev_emb': dev_emb,
                'after_layer': transformer_layer_index,
                'timestamp': str(datetime.datetime.now())
            }, f"{args.save_path}/dev_activations.pt")

        torch.cuda.empty_cache()
        print(f"Completed layer {transformer_layer_index}")

    print("\n" + "="*50)
    print("All layers processed successfully!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect Hessian matrices for Qwen2-MoE expert layers')
    parser.add_argument('--base_model', type=str, required=True,
                       help='HuggingFace model name or path')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to save Hessian matrices')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for forward pass')
    parser.add_argument('--devset_size', type=int, default=512,
                       help='Number of samples for calibration')
    parser.add_argument('--ctx_size', type=int, default=4096,
                       help='Context size')
    parser.add_argument('--sample_proc', type=int, default=1,
                       help='Number of processes for sampling')
    parser.add_argument('--num_experts', type=int, default=None,
                       help='Number of experts (auto-detected from model config if not specified)')
    parser.add_argument('--save_activations', action='store_true',
                       help='Save intermediate activations for resuming')
    parser.add_argument('--moe_only', action='store_true', default=True,
                       help='Only process MoE layers (default: True)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    main(args)