import argparse
import datetime
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from lib import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def forward_layer_qwen15moe(
    layer,
    position_ids,
    attention_mask,
    bs,
    device,
    dev_emb,
    num_experts
):
    """Forward pass for Qwen1.5-MoE layer - collecting Q/K/V/O projection Hessians."""
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
    # Check if this layer has MoE structure
    # if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
    #     done_gate = utils.register_H_hook(layer.mlp.gate, device)
    #     
    #     done_experts = {}
    #     for expert_idx in range(num_experts):
    #         expert = layer.mlp.experts[expert_idx]
    #         done_experts[f'expert{expert_idx}_gate_proj'] = utils.register_H_hook(expert.gate_proj, device)
    #         done_experts[f'expert{expert_idx}_up_proj'] = utils.register_H_hook(expert.up_proj, device)
    #         done_experts[f'expert{expert_idx}_down_proj'] = utils.register_H_hook(expert.down_proj, device)

    # Forward pass through layer
    assert len(dev_emb) % bs == 0
    for i in tqdm(range(len(dev_emb) // bs), desc="Forward pass"):
        # Create batch with correct device and shape
        batch = dev_emb[i * bs:(i + 1) * bs].to(device)
        batch_pos_ids = position_ids[:batch.shape[0]].to(device)
        batch_attn_mask = attention_mask[:batch.shape[0]] if attention_mask.shape[0] > batch.shape[0] else attention_mask
        
        with torch.no_grad():
            # Qwen1.5-MoE might not support output_router_logits parameter
            # Try with the parameter first, fallback without it
            try:
                output = layer(
                    batch,
                    position_ids=batch_pos_ids,
                    attention_mask=batch_attn_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_router_logits=False
                )[0]
            except TypeError:
                # If output_router_logits is not supported, call without it
                output = layer(
                    batch,
                    position_ids=batch_pos_ids,
                    attention_mask=batch_attn_mask,
                    use_cache=False,
                    output_attentions=False,
                )[0]
        
        # Use .copy_() to avoid in-place operation errors
        dev_emb[i * bs:(i + 1) * bs].copy_(output.cpu())

    # Collect results - Q/K/V/O projections
    layer = layer.cpu()
    
    results = {
        'q': done_q(),
        'k': done_k(),
        'v': done_v(),
        'o': done_o(),
    }
    
    # Expert layers (commented out - not collecting yet)
    # if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
    #     results['gate'] = done_gate()
    #     for key, done_fn in done_experts.items():
    #         results[key] = done_fn()

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


def count_linear_layers_qwen15moe(layer, num_experts):
    """Count expected linear layers in Qwen1.5-MoE - collecting Q/K/V/O projections."""
    # Counting: q_proj, k_proj, v_proj, o_proj = 4 layers
    # Not collecting: gate + experts*(gate_proj, up_proj, down_proj)
    return 4  # q, k, v, o


def detect_qwen15moe_config(model):
    """Detect Qwen1.5-MoE specific configuration."""
    config = {
        'num_experts': 8,  # default
        'has_shared_expert': False,
        'is_moe_layer': [],
        'model_type': 'unknown'
    }
    
    # Detect model type
    if hasattr(model.config, 'model_type'):
        config['model_type'] = model.config.model_type
        print(f"Detected model_type: {model.config.model_type}")
    
    # Try to get num_experts from config
    if hasattr(model.config, 'num_experts'):
        config['num_experts'] = model.config.num_experts
    elif hasattr(model.config, 'num_local_experts'):
        config['num_experts'] = model.config.num_local_experts
    elif hasattr(model.config, 'moe_num_experts'):
        config['num_experts'] = model.config.moe_num_experts
    
    # Check which layers are MoE layers and which are standard layers
    for i, layer in enumerate(model.model.layers):
        # Check if this layer has MoE structure
        if hasattr(layer, 'mlp'):
            # MoE layer: has gate and experts
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
                config['is_moe_layer'].append(True)
                # Check if this layer has shared expert
                if hasattr(layer.mlp, 'shared_expert') or hasattr(layer.mlp, 'shared_expert_gate'):
                    config['has_shared_expert'] = True
                print(f"Layer {i}: MoE layer detected")
            # Standard FFN layer: has gate_proj, up_proj, down_proj directly
            elif hasattr(layer.mlp, 'gate_proj'):
                config['is_moe_layer'].append(False)
                print(f"Layer {i}: Standard FFN layer detected")
            else:
                config['is_moe_layer'].append(False)
                print(f"Layer {i}: Unknown MLP structure")
        else:
            config['is_moe_layer'].append(False)
            print(f"Layer {i}: No mlp attribute found")
    
    return config


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True  # Qwen models require trust_remote_code
    )
    print("Loaded model!")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        use_fast=True,
        trust_remote_code=True  # Qwen models require trust_remote_code
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Detect Qwen1.5-MoE configuration
    qwen_config = detect_qwen15moe_config(model)
    
    # Override num_experts if detected from model config
    if args.num_experts is None:
        args.num_experts = qwen_config['num_experts']
    
    print(f"\n{'='*50}")
    print(f"Model Configuration:")
    print(f"  Model type: {qwen_config['model_type']}")
    print(f"  Number of experts: {args.num_experts}")
    print(f"  Has shared expert: {qwen_config['has_shared_expert']}")
    print(f"  Total layers: {len(qwen_config['is_moe_layer'])}")
    print(f"  MoE layers: {sum(qwen_config['is_moe_layer'])}")
    print(f"  Standard layers: {len(qwen_config['is_moe_layer']) - sum(qwen_config['is_moe_layer'])}")
    print(f"{'='*50}\n")

    # Load or create dataset
    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("Loading cached dataset...")
        loaded_dev_activations = torch.load(f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(f"Loaded cached dataset from {loaded_dev_activations['timestamp']}")
        print(f"Resuming from layer {after_layer + 1}")
    else:
        print("Loading dataset...")
        devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                                   nproc=args.sample_proc)
        dev_emb = model.model.embed_tokens(devset).detach()  # Add .detach() to avoid gradient tracking
        after_layer = -1
        print("Loaded dataset!")
    
    # Ensure dev_emb doesn't require gradients
    dev_emb.requires_grad_(False)

    print(f"dev_emb shape: {dev_emb.shape}, dtype: {dev_emb.dtype}")

    # Prepare position ids and attention mask
    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :].expand(args.batch_size, -1).contiguous()

    if hasattr(model.config, 'sliding_window') and model.config.sliding_window is not None:
        print(f"Using sliding window attention with window size: {model.config.sliding_window}")
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
        
        # Print layer type
        if transformer_layer_index < len(qwen_config['is_moe_layer']):
            layer_type = "MoE" if qwen_config['is_moe_layer'][transformer_layer_index] else "Standard FFN"
            print(f"Layer type: {layer_type}")
        
        print(f"{'='*50}")

        transformer_layer = model.model.layers[transformer_layer_index]

        # Validate layer structure - all layers should have self_attn
        assert hasattr(transformer_layer, 'self_attn'), \
            f"Layer {transformer_layer_index} does not have 'self_attn' attribute"
        assert hasattr(transformer_layer.self_attn, 'q_proj'), \
            f"Layer {transformer_layer_index} self_attn does not have 'q_proj'"
        assert hasattr(transformer_layer.self_attn, 'k_proj'), \
            f"Layer {transformer_layer_index} self_attn does not have 'k_proj'"
        assert hasattr(transformer_layer.self_attn, 'v_proj'), \
            f"Layer {transformer_layer_index} self_attn does not have 'v_proj'"
        assert hasattr(transformer_layer.self_attn, 'o_proj'), \
            f"Layer {transformer_layer_index} self_attn does not have 'o_proj'"
        
        # Count linear layers for validation
        linear_count = len([m for m in transformer_layer.modules()
                           if isinstance(m, torch.nn.Linear)])
        expected_count = count_linear_layers_qwen15moe(transformer_layer, args.num_experts)
        print(f"Found {linear_count} linear layers, collecting Hessians for {expected_count} layers (Q/K/V/O)")

        # Forward pass and collect Hessians
        print("Running forward pass to collect Hessians...")
        results = forward_layer_qwen15moe(
            transformer_layer,
            position_ids,
            attention_mask,
            args.batch_size,
            device,
            dev_emb,
            args.num_experts
        )

        # Save Hessians
        print("Saving Hessians...")
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
    parser = argparse.ArgumentParser(description='Collect Q/K/V/O Hessians for Qwen1.5-MoE model')

    parser.add_argument('--seed', default=0, type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--batch_size', default=2, type=int,
                       help='Batch size for forward pass')
    parser.add_argument('--devset_size', default=256, type=int,
                       help='Number of samples for calibration dataset')
    parser.add_argument('--ctx_size', default=4096, type=int,
                       help='Context size (sequence length)')
    parser.add_argument('--base_model', default='Qwen/Qwen1.5-MoE-A2.7B', type=str,
                       help='HuggingFace model name or path')
    parser.add_argument('--save_path', default='hessians/qwen15_moe', type=str,
                       help='Path to save Hessian matrices')
    parser.add_argument('--act_save_rate', default=4, type=int,
                       help='Save activation checkpoint every N layers')
    parser.add_argument('--save_activations', action='store_true',
                       help='Save intermediate activations for resuming')
    parser.add_argument('--sample_proc', default=8, type=int,
                       help='Number of processes for sampling dataset')
    parser.add_argument('--num_experts', default=None, type=int,
                       help='Number of experts (auto-detected from model config if not specified)')

    torch.set_grad_enabled(False)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)
    main(args)