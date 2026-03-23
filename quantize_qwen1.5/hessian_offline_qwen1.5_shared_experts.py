import argparse
import datetime
import os
import sys

# Add parent directory to path to allow importing lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from lib import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def verify_shared_expert_count(model):
    """
    详细检查第一个MoE层的Shared Expert结构
    """
    print("\n" + "="*70)
    print("DETAILED SHARED EXPERT STRUCTURE ANALYSIS")
    print("="*70)
    
    for i, layer in enumerate(model.model.layers):
        if not hasattr(layer, 'mlp'):
            continue
        
        moe_module = layer.mlp
        
        # 检查是否是MoE层
        is_moe = hasattr(moe_module, 'gate') and hasattr(moe_module, 'experts')
        
        if not is_moe:
            continue
        
        print(f"\nLayer {i} - MoE Module Attributes:")
        print("-" * 70)
        
        # 列出所有与shared相关的属性
        shared_attrs = [attr for attr in dir(moe_module) if 'shared' in attr.lower() and not attr.startswith('_')]
        
        if not shared_attrs:
            print("❌ No shared-related attributes found!")
            continue
        
        print(f"Found {len(shared_attrs)} shared-related attributes: {shared_attrs}\n")
        
        for attr_name in shared_attrs:
            attr = getattr(moe_module, attr_name)
            print(f"📌 {attr_name}:")
            print(f"   Type: {type(attr).__name__}")
            
            # 如果是列表或ModuleList
            if isinstance(attr, (list, torch.nn.ModuleList)):
                print(f"   ✅ This is a LIST/ModuleList")
                print(f"   Length: {len(attr)}")
                if len(attr) > 0:
                    print(f"   First element type: {type(attr[0]).__name__}")
                    
                    # 检查第一个元素的子模块
                    first_expert = attr[0]
                    if hasattr(first_expert, 'gate_proj'):
                        print(f"   Structure of each expert:")
                        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                            if hasattr(first_expert, proj_name):
                                proj = getattr(first_expert, proj_name)
                                print(f"     - {proj_name}: {proj.weight.shape}")
            
            # 如果是单个模块
            elif isinstance(attr, torch.nn.Module):
                print(f"   ✅ This is a SINGLE Module")
                
                # 检查是否有projection layers
                has_projections = any(hasattr(attr, name) for name in ['gate_proj', 'up_proj', 'down_proj'])
                
                if has_projections:
                    print(f"   Structure:")
                    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(attr, proj_name):
                            proj = getattr(attr, proj_name)
                            print(f"     - {proj_name}: {proj.weight.shape}")
                else:
                    # 可能是gate layer
                    if hasattr(attr, 'weight'):
                        print(f"     - weight: {attr.weight.shape}")
            
            else:
                print(f"   ⚠️  Unknown type: {type(attr)}")
            
            print()
        
        # 只分析第一个MoE层
        print("="*70)
        print("(Only showing first MoE layer for analysis)\n")
        break
    
    input("Press Enter to continue with Hessian collection...")


def forward_layer_all_shared_experts(
    layer,
    position_ids,
    attention_mask,
    bs,
    device,
    dev_emb
):
    """
    Forward pass for Qwen1.5-MoE layer - collecting ALL Shared Experts' Hessians.
    
    Automatically detects:
    - shared_expert (single module)
    - shared_expert (ModuleList/List)
    - shared_experts (ModuleList/List)
    """
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)

    moe_module = layer.mlp
    
    print("\nCollecting Hessians for Shared Expert components...")
    
    done_shared = {}
    num_shared_experts = 0
    
    # ===== 检测并注册 Shared Expert(s) =====
    
    # 情况1: shared_expert (可能是单个或列表)
    if hasattr(moe_module, 'shared_expert'):
        shared_attr = moe_module.shared_expert
        
        # 子情况1a: shared_expert是ModuleList或list
        if isinstance(shared_attr, (torch.nn.ModuleList, list)):
            num_shared_experts = len(shared_attr)
            print(f"✅ Found shared_expert as LIST with {num_shared_experts} experts")
            
            for expert_idx, shared_expert in enumerate(shared_attr):
                # 为每个shared expert的3个投影层注册hook
                if hasattr(shared_expert, 'gate_proj'):
                    done_shared[f'shared_expert_{expert_idx}_gate_proj'] = \
                        utils.register_H_hook(shared_expert.gate_proj, device)
                    print(f"  - Registered: shared_expert_{expert_idx}.gate_proj")
                
                if hasattr(shared_expert, 'up_proj'):
                    done_shared[f'shared_expert_{expert_idx}_up_proj'] = \
                        utils.register_H_hook(shared_expert.up_proj, device)
                    print(f"  - Registered: shared_expert_{expert_idx}.up_proj")
                
                if hasattr(shared_expert, 'down_proj'):
                    done_shared[f'shared_expert_{expert_idx}_down_proj'] = \
                        utils.register_H_hook(shared_expert.down_proj, device)
                    print(f"  - Registered: shared_expert_{expert_idx}.down_proj")
        
        # 子情况1b: shared_expert是单个模块
        elif isinstance(shared_attr, torch.nn.Module):
            num_shared_experts = 1
            print(f"✅ Found shared_expert as SINGLE module")
            shared_expert = shared_attr
            
            if hasattr(shared_expert, 'gate_proj'):
                done_shared['shared_expert_0_gate_proj'] = \
                    utils.register_H_hook(shared_expert.gate_proj, device)
                print("  - Registered: shared_expert_0.gate_proj")
            
            if hasattr(shared_expert, 'up_proj'):
                done_shared['shared_expert_0_up_proj'] = \
                    utils.register_H_hook(shared_expert.up_proj, device)
                print("  - Registered: shared_expert_0.up_proj")
            
            if hasattr(shared_expert, 'down_proj'):
                done_shared['shared_expert_0_down_proj'] = \
                    utils.register_H_hook(shared_expert.down_proj, device)
                print("  - Registered: shared_expert_0.down_proj")
    
    # 情况2: shared_experts (复数，通常是列表)
    elif hasattr(moe_module, 'shared_experts'):
        shared_experts = moe_module.shared_experts
        
        if isinstance(shared_experts, (torch.nn.ModuleList, list)):
            num_shared_experts = len(shared_experts)
            print(f"✅ Found shared_experts (plural) with {num_shared_experts} experts")
            
            for expert_idx, shared_expert in enumerate(shared_experts):
                if hasattr(shared_expert, 'gate_proj'):
                    done_shared[f'shared_expert_{expert_idx}_gate_proj'] = \
                        utils.register_H_hook(shared_expert.gate_proj, device)
                    print(f"  - Registered: shared_expert_{expert_idx}.gate_proj")
                
                if hasattr(shared_expert, 'up_proj'):
                    done_shared[f'shared_expert_{expert_idx}_up_proj'] = \
                        utils.register_H_hook(shared_expert.up_proj, device)
                    print(f"  - Registered: shared_expert_{expert_idx}.up_proj")
                
                if hasattr(shared_expert, 'down_proj'):
                    done_shared[f'shared_expert_{expert_idx}_down_proj'] = \
                        utils.register_H_hook(shared_expert.down_proj, device)
                    print(f"  - Registered: shared_expert_{expert_idx}.down_proj")
        else:
            raise ValueError(f"shared_experts has unexpected type: {type(shared_experts)}")
    
    else:
        raise ValueError(
            "No shared_expert or shared_experts attribute found in MoE module!\n"
            "Available attributes: " + str([attr for attr in dir(moe_module) if not attr.startswith('_')])
        )
    
    # ===== 注册 Shared Expert Gate =====
    if hasattr(moe_module, 'shared_expert_gate'):
        done_shared['shared_expert_gate_layer'] = utils.register_H_hook(
            moe_module.shared_expert_gate, device
        )
        print("  - Registered: shared_expert_gate")
    else:
        print("  ⚠️  Warning: No shared_expert_gate found")
    
    print(f"\n✅ Total hooks registered: {len(done_shared)}")
    print(f"   Expected per layer: {num_shared_experts * 3 + 1} (3 projections × {num_shared_experts} experts + 1 gate)")

    # ===== Forward Pass =====
    assert len(dev_emb) % bs == 0
    num_batches = len(dev_emb) // bs
    
    for i in tqdm(range(num_batches), desc="Forward pass", leave=False):
        # Create batch
        batch = dev_emb[i * bs:(i + 1) * bs].to(device)
        batch_pos_ids = position_ids[:batch.shape[0]].to(device)
        
        # Handle attention mask
        if attention_mask.shape[0] > batch.shape[0]:
            batch_attn_mask = attention_mask[:batch.shape[0]]
        else:
            batch_attn_mask = attention_mask
        
        # Forward pass
        with torch.no_grad():
            output = layer(
                batch,
                position_ids=batch_pos_ids,
                attention_mask=batch_attn_mask,
                use_cache=False,
                output_attentions=False,
            )[0]
        
        # Update embeddings (avoid in-place operation)
        dev_emb[i * bs:(i + 1) * bs].copy_(output.cpu())

    # ===== Collect Results =====
    layer = layer.cpu()
    
    results = {}
    for key, done_fn in done_shared.items():
        results[key] = done_fn()
    
    print(f"✅ Collected {len(results)} Hessian matrices")
    
    return results, num_shared_experts


def save_hessians(results, args, transformer_layer_index):
    """Process and save Hessians for shared expert components."""
    print(f"\nProcessing and saving Hessians for layer {transformer_layer_index}...")
    
    for key, (H, mu, ct) in results.items():
        print(f"  Processing {key}...")
        print(f"    Matrix shape: {H.shape}, Count: {ct}")
        
        # Compute covariance: E[xx^T] - E[x]E[x]^T
        mu = mu / ct
        H = H / ct
        # Subtract outer product: H = H - mu @ mu^T
        H.sub_(mu.unsqueeze(-1) @ mu.unsqueeze(0))
        
        # Symmetrize the matrix to ensure it's perfectly symmetric
        H = (H + H.T) / 2
        
        # Ensure positive semi-definiteness
        try:
            eigvals = torch.linalg.eigvalsh(H)
            min_eigval = eigvals.min().item()
            max_eigval = eigvals.max().item()
            
            print(f"    Eigenvalues: min={min_eigval:.6f}, max={max_eigval:.6f}")
            
            if min_eigval < 0:
                # Add small regularization to ensure positive semi-definiteness
                regularization = abs(min_eigval) * 1.1
                H.add_(torch.eye(H.shape[0], dtype=H.dtype, device=H.device) * regularization)
                print(f"    Applied regularization: {regularization:.6f}")
        except Exception as e:
            print(f"    Warning: Eigenvalue computation failed: {e}")

        # Save path
        save_path = f"{args.save_path}/{transformer_layer_index}_{key}.pt"

        # Save
        torch.save({
            'flatH': utils.sym_to_flat(H.to(torch.float32)),
            'mu': mu.to(torch.float32),
            'n': H.shape[0],
            'ct': ct
        }, save_path)

        print(f"    ✅ Saved to {save_path}")

    del results


def verify_shared_expert_structure(model):
    """
    Verify that the model has shared expert structure.
    Returns (has_shared_expert, moe_layer_indices, num_shared_experts_per_layer).
    """
    moe_layers_with_shared = []
    num_shared_experts = None
    
    for i, layer in enumerate(model.model.layers):
        if not hasattr(layer, 'mlp'):
            continue
        
        moe_module = layer.mlp
        
        # Check if this is a MoE layer
        is_moe = hasattr(moe_module, 'gate') and hasattr(moe_module, 'experts')
        
        if not is_moe:
            continue
        
        # Check shared expert(s)
        has_shared = False
        layer_num_shared = 0
        
        # Check for shared_expert (singular)
        if hasattr(moe_module, 'shared_expert'):
            shared_attr = moe_module.shared_expert
            if isinstance(shared_attr, (torch.nn.ModuleList, list)):
                layer_num_shared = len(shared_attr)
            elif isinstance(shared_attr, torch.nn.Module):
                layer_num_shared = 1
            else:
                print(f"⚠️  Layer {i}: shared_expert has unexpected type {type(shared_attr)}")
                continue
            has_shared = True
        
        # Check for shared_experts (plural)
        elif hasattr(moe_module, 'shared_experts'):
            shared_experts = moe_module.shared_experts
            if isinstance(shared_experts, (torch.nn.ModuleList, list)):
                layer_num_shared = len(shared_experts)
            else:
                print(f"⚠️  Layer {i}: shared_experts has unexpected type {type(shared_experts)}")
                continue
            has_shared = True
        
        if is_moe and has_shared:
            moe_layers_with_shared.append(i)
            
            # Verify all layers have the same number of shared experts
            if num_shared_experts is None:
                num_shared_experts = layer_num_shared
            elif num_shared_experts != layer_num_shared:
                raise ValueError(
                    f"Inconsistent number of shared experts! "
                    f"Layer {i} has {layer_num_shared}, but previous layers have {num_shared_experts}"
                )
            
            # Print structure info for first layer
            if len(moe_layers_with_shared) == 1:
                print(f"\n{'='*70}")
                print(f"Shared Expert Structure (Layer {i}):")
                print(f"{'='*70}")
                print(f"Number of shared experts: {layer_num_shared}")
                
                # Get list of experts to display
                if hasattr(moe_module, 'shared_expert'):
                    shared_attr = moe_module.shared_expert
                    if isinstance(shared_attr, (list, torch.nn.ModuleList)):
                        experts_to_show = shared_attr
                    else:
                        experts_to_show = [shared_attr]
                elif hasattr(moe_module, 'shared_experts'):
                    experts_to_show = moe_module.shared_experts
                
                # Display structure of each expert
                total_params = 0
                for expert_idx, expert in enumerate(experts_to_show):
                    print(f"\n  Shared Expert {expert_idx}:")
                    print(f"    Type: {type(expert).__name__}")
                    
                    expert_params = 0
                    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(expert, proj_name):
                            proj = getattr(expert, proj_name)
                            shape = proj.weight.shape
                            params = proj.weight.numel()
                            expert_params += params
                            print(f"      - {proj_name}: {shape} ({params:,} params)")
                    
                    total_params += expert_params
                    print(f"    Subtotal: {expert_params:,} parameters")
                
                # Display shared expert gate
                if hasattr(moe_module, 'shared_expert_gate'):
                    gate = moe_module.shared_expert_gate
                    gate_params = gate.weight.numel()
                    total_params += gate_params
                    print(f"\n  Shared Expert Gate:")
                    print(f"    Type: {type(gate).__name__}")
                    print(f"    Weight: {gate.weight.shape} ({gate_params:,} params)")
                
                print(f"\n  Total Shared Expert Parameters: {total_params:,}")
                print(f"{'='*70}\n")
    
    return len(moe_layers_with_shared) > 0, moe_layers_with_shared, num_shared_experts


def check_existing_hessians(save_path, layer_indices, num_shared_experts):
    """
    Check which layers already have saved Hessians.
    
    Args:
        save_path: Directory where Hessians are saved
        layer_indices: List of layer indices to check
        num_shared_experts: Number of shared experts per layer
    """
    completed_layers = []
    
    for layer_idx in layer_indices:
        # Check if all required Hessian files exist
        required_files = []
        
        # Each shared expert has 3 projection layers
        for expert_idx in range(num_shared_experts):
            required_files.extend([
                f"{layer_idx}_shared_expert_{expert_idx}_gate_proj.pt",
                f"{layer_idx}_shared_expert_{expert_idx}_up_proj.pt",
                f"{layer_idx}_shared_expert_{expert_idx}_down_proj.pt",
            ])
        
        # Add shared_expert_gate
        required_files.append(f"{layer_idx}_shared_expert_gate_layer.pt")
        
        all_exist = all(
            os.path.exists(os.path.join(save_path, f)) 
            for f in required_files
        )
        
        if all_exist:
            completed_layers.append(layer_idx)
    
    return completed_layers


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    print("\n" + "="*70)
    print("Qwen1.5-MoE Shared Expert Hessian Collection")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✅ Model loaded!")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        use_fast=True, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detailed structure analysis (if requested)
    if args.analyze_structure:
        verify_shared_expert_count(model)

    # Verify shared expert structure
    print("\nVerifying shared expert structure...")
    has_shared, moe_layers, num_shared_experts = verify_shared_expert_structure(model)
    
    if not has_shared:
        raise ValueError(
            "Model does not have shared expert structure! "
            "This script is only for Qwen1.5-MoE models with shared experts."
        )
    
    print(f"✅ Found {len(moe_layers)} MoE layers with shared experts")
    print(f"✅ Each layer has {num_shared_experts} shared expert(s)")
    print(f"   MoE Layer indices: {moe_layers}")
    print(f"   Expected Hessians per layer: {num_shared_experts * 3 + 1}")
    print(f"     ({num_shared_experts} experts × 3 projections + 1 gate)")
    
    # Check existing Hessians
    if not args.force:
        completed = check_existing_hessians(args.save_path, moe_layers, num_shared_experts)
        if completed:
            print(f"\n⚠️  Found existing Hessians for {len(completed)} layers: {completed}")
            print("   Use --force to recompute, or they will be skipped")
            remaining = [l for l in moe_layers if l not in completed]
            if remaining:
                print(f"   Remaining layers to process: {remaining}")
            else:
                print("   ✅ All layers already completed!")
                return
    else:
        completed = []

    # Load or create dataset
    cache_path = f"{args.save_path}/shared_expert_dev_activations.pt"
    
    if os.path.isfile(cache_path) and not args.force:
        print(f"\nLoading cached dataset from {cache_path}...")
        loaded_dev_activations = torch.load(cache_path)
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(f"✅ Loaded cached dataset from {loaded_dev_activations['timestamp']}")
        print(f"   Resuming after layer {after_layer}")
    else:
        print(f"\nCreating calibration dataset...")
        print(f"  - Dataset size: {args.devset_size}")
        print(f"  - Context size: {args.ctx_size}")
        print(f"  - Processes: {args.sample_proc}")
        
        devset = utils.sample_rp1t(
            tokenizer, 
            args.devset_size, 
            args.ctx_size,
            nproc=args.sample_proc
        )
        dev_emb = model.model.embed_tokens(devset).detach()
        after_layer = -1
        print("✅ Dataset created!")
    
    # Ensure dev_emb doesn't require gradients
    dev_emb.requires_grad_(False)
    print(f"\nDataset info:")
    print(f"  - Shape: {dev_emb.shape}")
    print(f"  - Dtype: {dev_emb.dtype}")
    print(f"  - Device: {dev_emb.device}")

    # Prepare position ids and attention mask
    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :].expand(
        args.batch_size, -1
    ).contiguous()

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

    print(f"\nAttention mask shape: {attention_mask.shape}")

    # Process each MoE layer
    print(f"\n{'='*70}")
    print(f"Processing {len(moe_layers)} MoE layers")
    print(f"{'='*70}\n")
    
    processed_count = 0
    skipped_count = 0
    
    for idx, layer_idx in enumerate(moe_layers):
        # Skip if already in cache
        if layer_idx <= after_layer:
            print(f"[{idx+1}/{len(moe_layers)}] Skipping layer {layer_idx} (already in activation cache)")
            skipped_count += 1
            continue
        
        # Skip if already completed and not forcing
        if not args.force and layer_idx in completed:
            print(f"[{idx+1}/{len(moe_layers)}] Skipping layer {layer_idx} (Hessians already exist)")
            skipped_count += 1
            continue

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(moe_layers)}] Processing Layer {layer_idx}")
        print(f"{'='*70}")

        transformer_layer = model.model.layers[layer_idx]

        # Verify layer has shared expert
        if not (hasattr(transformer_layer.mlp, 'shared_expert') or 
                hasattr(transformer_layer.mlp, 'shared_experts')):
            print(f"⚠️  Layer {layer_idx} does not have shared expert(s), skipping")
            skipped_count += 1
            continue

        # Forward pass and collect Hessians
        print("Running forward pass to collect Hessians...")
        try:
            results, layer_num_shared = forward_layer_all_shared_experts(
                transformer_layer,
                position_ids,
                attention_mask,
                args.batch_size,
                device,
                dev_emb
            )
            
            # Verify number of shared experts matches
            if layer_num_shared != num_shared_experts:
                print(f"⚠️  Warning: Layer {layer_idx} has {layer_num_shared} shared experts, "
                      f"but expected {num_shared_experts}")
            
        except Exception as e:
            print(f"❌ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

        # Verify we collected all expected Hessians
        expected_count = num_shared_experts * 3 + 1  # 3 projections per expert + 1 gate
        collected_count = len(results)
        
        if collected_count != expected_count:
            print(f"⚠️  Warning: Collected {collected_count} Hessians, expected {expected_count}")
            
            # Show what we have
            collected_keys = set(results.keys())
            print(f"   Collected keys: {sorted(collected_keys)}")
        
        # Save Hessians
        print("\nSaving Hessians...")
        try:
            save_hessians(results, args, layer_idx)
            processed_count += 1
        except Exception as e:
            print(f"❌ Error saving Hessians: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

        # Save activations if requested
        if args.save_activations:
            print("\nSaving activations...")
            torch.save({
                'dev_emb': dev_emb,
                'after_layer': layer_idx,
                'timestamp': str(datetime.datetime.now())
            }, cache_path)
            print(f"✅ Saved activations to {cache_path}")

        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        print(f"✅ Completed layer {layer_idx}")

    print("\n" + "="*70)
    print("🎉 Hessian collection finished!")
    print("="*70)
    
    # Summary
    print("\nSummary:")
    print(f"  - Total MoE layers: {len(moe_layers)}")
    print(f"  - Layers processed: {processed_count}")
    print(f"  - Layers skipped: {skipped_count}")
    print(f"  - Shared experts per layer: {num_shared_experts}")
    print(f"  - Hessians per layer: {num_shared_experts * 3 + 1}")
    print(f"  - Total Hessian files created: {processed_count * (num_shared_experts * 3 + 1)}")
    print(f"  - Save path: {args.save_path}")
    
    # Check completeness
    final_completed = check_existing_hessians(args.save_path, moe_layers, num_shared_experts)
    if len(final_completed) == len(moe_layers):
        print("\n✅ All layers have complete Hessian files!")
    else:
        missing = [l for l in moe_layers if l not in final_completed]
        print(f"\n⚠️  Warning: {len(missing)} layers missing Hessians: {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect Hessian matrices for ALL Qwen1.5-MoE Shared Experts'
    )
    
    # Required arguments
    parser.add_argument(
        '--base_model', 
        type=str, 
        required=True,
        help='HuggingFace model name or path (e.g., Qwen/Qwen1.5-MoE-A2.7B)'
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        required=True,
        help='Path to save Hessian matrices'
    )
    
    # Optional arguments
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8,
        help='Batch size for forward pass (default: 8)'
    )
    parser.add_argument(
        '--devset_size', 
        type=int, 
        default=512,
        help='Number of samples for calibration (default: 512)'
    )
    parser.add_argument(
        '--ctx_size', 
        type=int, 
        default=4096,
        help='Context size (default: 4096)'
    )
    parser.add_argument(
        '--sample_proc', 
        type=int, 
        default=1,
        help='Number of processes for sampling (default: 1)'
    )
    parser.add_argument(
        '--save_activations', 
        action='store_true',
        help='Save intermediate activations for resuming'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force recomputation even if Hessians already exist'
    )
    parser.add_argument(
        '--analyze_structure',
        action='store_true',
        help='Run detailed structure analysis before processing (interactive)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"  Model: {args.base_model}")
    print(f"  Save path: {args.save_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dataset size: {args.devset_size}")
    print(f"  Context size: {args.ctx_size}")
    print(f"  Sample processes: {args.sample_proc}")
    print(f"  Save activations: {args.save_activations}")
    print(f"  Force recompute: {args.force}")
    print(f"  Analyze structure: {args.analyze_structure}")
    print("="*70 + "\n")
    
    main(args)