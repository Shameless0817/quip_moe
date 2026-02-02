import argparse
import datetime
import os
import random
import sys
from copy import deepcopy

from tqdm import tqdm

# 添加父目录到Python路径，以便能够导入lib模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
import torch.multiprocessing as mp
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerFast)
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--devset_size', default=256, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model',
                    default='mistralai/Mixtral-8x7B-v0.1',
                    type=str)
parser.add_argument('--save_path', default='hessians/mixtral_8x7b', type=str)
parser.add_argument('--scratch_path', default=None, type=str)
parser.add_argument('--chunk_size', default=256, type=int)
parser.add_argument('--async_copy_speed', default=-1, type=int)
parser.add_argument('--act_save_rate', default=4, type=int)
parser.add_argument('--save_activations', action='store_true')
parser.add_argument('--sample_proc', default=4, type=int)
parser.add_argument('--num_experts', default=8, type=int)


def move_fn(in_q, async_copy_speed):
    """Async copy to avoid slow disk"""
    while True:
        item = in_q.get()
        if item is None:
            return
        src, tgt = item
        if async_copy_speed > 0:
            os.system(f'rsync --bwlimit={async_copy_speed} {src} {tgt}')
        else:
            os.system(f'rsync {src} {tgt}')
        os.system(f'rm {src}')
        print(f'moved {src} to {tgt}')


def forward_layer_mixtral(layer, position_ids, attention_mask, bs, device,
                          in_q, out_q, result_q, num_experts):
    """
    Forward pass for Mixtral layer with MoE support.
    Mixtral uses:
    - self_attn: q_proj, k_proj, v_proj, o_proj
    - block_sparse_moe: gate + experts (each expert has w1, w2, w3)
    
    Args:
        layer: The transformer layer to process
        position_ids: Position IDs tensor
        attention_mask: Attention mask tensor
        bs: Batch size
        device: GPU device index
        in_q: Input queue for receiving chunks
        out_q: Output queue for Hessian results
        result_q: Output queue for processed activations
        num_experts: Number of experts in the MoE layer
    """
    torch.set_grad_enabled(False)
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Register hooks for attention layers
    # Mixtral has separate q, k, v projections
    done_q = utils.register_H_hook(layer.self_attn.q_proj, device)
    done_k = utils.register_H_hook(layer.self_attn.k_proj, device)
    done_v = utils.register_H_hook(layer.self_attn.v_proj, device)
    done_o = utils.register_H_hook(layer.self_attn.o_proj, device)

    # Register hooks for MoE gate
    done_gate = utils.register_H_hook(layer.block_sparse_moe.gate, device)

    # Register hooks for each expert's layers
    # Each expert has: w1 (gate_proj), w2 (down_proj), w3 (up_proj)
    done_experts = {}
    for expert_idx in range(num_experts):
        expert = layer.block_sparse_moe.experts[expert_idx]
        done_experts[f'expert{expert_idx}_w1'] = utils.register_H_hook(expert.w1, device)
        done_experts[f'expert{expert_idx}_w2'] = utils.register_H_hook(expert.w2, device)
        done_experts[f'expert{expert_idx}_w3'] = utils.register_H_hook(expert.w3, device)

    while True:
        item = in_q.get()
        if item is None:
            # Cleanup and return Hessian results
            layer = layer.cpu()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()

            # Collect all Hessian results
            results = {
                'q': done_q(),
                'k': done_k(),
                'v': done_v(),
                'o': done_o(),
                'gate': done_gate(),
            }

            # Add expert results
            for key, done_fn in done_experts.items():
                results[key] = done_fn()

            out_q.put(results)
            return

        chunk_idx, chunk_data = item
        chunk_data = chunk_data.to(device)
        
        assert len(chunk_data) % bs == 0, \
            f"Chunk size {len(chunk_data)} not divisible by batch size {bs}"
        
        processed_batches = []
        for i in range(len(chunk_data) // bs):
            batch_input = chunk_data[i * bs:(i + 1) * bs]
            batch_output = layer(
                batch_input,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_router_logits=False)[0]
            processed_batches.append(batch_output.cpu())
        
        # Send processed chunk back with its index
        processed_chunk = torch.cat(processed_batches, dim=0)
        result_q.put((chunk_idx, processed_chunk))


def accumulate_mixtral(in_q, move_q, ngpus, args, transformer_layer_index):
    """Accumulate Hessians from all GPUs for Mixtral"""
    Hs = {}
    mus = {}
    cts = {}

    for i in range(ngpus):
        out = in_q.get()
        if i == 0:
            for key in out:
                Hs[key] = torch.zeros(out[key][0].shape,
                                      dtype=out[key][0].dtype)
                mus[key] = torch.zeros(out[key][1].shape,
                                       dtype=out[key][1].dtype)
                cts[key] = 0
        for key in out:
            Hs[key].add_(out[key][0])
            mus[key].add_(out[key][1])
            cts[key] += out[key][2]

    for key in Hs:
        # Handle case where expert was never activated
        if cts[key] == 0:
            print(f"Warning: {key} was never activated, skipping Hessian computation")
            continue
            
        mus[key].div_(cts[key])
        Hs[key].div_(cts[key])
        Hs[key].addmm_(-mus[key].unsqueeze(-1), mus[key].unsqueeze(0))

        save_path = f"{args.scratch_path}/{transformer_layer_index}_{key}.pt" \
            if args.scratch_path is not None \
            else f"{args.save_path}/{transformer_layer_index}_{key}.pt"

        torch.save(
            {
                'flatH': utils.sym_to_flat(Hs[key].to(torch.float32)),
                'mu': mus[key].to(torch.float32),
                'n': Hs[key].shape[0],
                'ct': cts[key]
            }, save_path)

        if args.scratch_path is not None:
            move_q.put(
                (f"{args.scratch_path}/{transformer_layer_index}_{key}.pt",
                 f"{args.save_path}/{transformer_layer_index}_{key}.pt"))

    del Hs, mus, cts, out


def collect_results(result_q, dev_emb, total_chunks, lock):
    """
    Collect processed chunks from workers and update dev_emb in-place.
    
    Args:
        result_q: Queue containing (chunk_idx, processed_data) tuples
        dev_emb: Shared memory tensor to update
        total_chunks: Total number of chunks to collect
        lock: Lock for synchronizing writes to dev_emb
    """
    collected = 0
    chunk_results = {}
    
    while collected < total_chunks:
        chunk_idx, processed_chunk = result_q.get()
        chunk_results[chunk_idx] = processed_chunk
        collected += 1
    
    # Sort and update dev_emb in order
    for chunk_idx in sorted(chunk_results.keys()):
        start_idx, end_idx = chunk_idx
        with lock:
            dev_emb[start_idx:end_idx].copy_(chunk_results[chunk_idx])
    
    del chunk_results


def count_linear_layers_mixtral(layer, num_experts):
    """
    Count expected linear layers in Mixtral:
    - Attention: q_proj, k_proj, v_proj, o_proj (4 layers)
    - MoE gate: 1 layer
    - Experts: num_experts * 3 (w1, w2, w3 per expert)
    Total: 4 + 1 + num_experts * 3
    """
    return 4 + 1 + num_experts * 3


def verify_layer_structure(transformer_layer, num_experts, layer_idx):
    """Verify that the layer has the expected Mixtral structure."""
    # Check for MoE structure
    if not hasattr(transformer_layer, 'block_sparse_moe'):
        raise ValueError(f"Layer {layer_idx} does not have block_sparse_moe - not a Mixtral model?")
    
    if not hasattr(transformer_layer.block_sparse_moe, 'experts'):
        raise ValueError(f"Layer {layer_idx} MoE layer does not have experts attribute")
    
    actual_experts = len(transformer_layer.block_sparse_moe.experts)
    if actual_experts != num_experts:
        raise ValueError(f"Layer {layer_idx}: Expected {num_experts} experts, found {actual_experts}")
    
    # Check attention structure
    if not hasattr(transformer_layer, 'self_attn'):
        raise ValueError(f"Layer {layer_idx} does not have self_attn")
    
    attn = transformer_layer.self_attn
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if not hasattr(attn, proj):
            raise ValueError(f"Layer {layer_idx} self_attn missing {proj}")
    
    # Check expert structure
    for expert_idx in range(num_experts):
        expert = transformer_layer.block_sparse_moe.experts[expert_idx]
        for w in ['w1', 'w2', 'w3']:
            if not hasattr(expert, w):
                raise ValueError(f"Layer {layer_idx} expert {expert_idx} missing {w}")
    
    # Count and verify linear layers
    linear_count = len([
        m for m in transformer_layer.modules()
        if isinstance(m, torch.nn.Linear)
    ])
    expected_count = count_linear_layers_mixtral(transformer_layer, num_experts)
    
    if linear_count != expected_count:
        print(f"Warning: Layer {layer_idx} has {linear_count} linear layers, expected {expected_count}")
    
    return True


def distribute_chunks_to_gpus(dev_emb, chunk_size, ngpus):
    """
    Create a list of chunk assignments for each GPU.
    Returns a list of (start_idx, end_idx) tuples.
    """
    chunks = []
    i = 0
    while i < len(dev_emb):
        end_idx = min(i + chunk_size, len(dev_emb))
        chunks.append((i, end_idx))
        i = end_idx
    return chunks


def main(args):
    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("loaded model!")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Get number of experts from model config
    if hasattr(model.config, 'num_local_experts'):
        args.num_experts = model.config.num_local_experts
        print(f"Detected {args.num_experts} experts from model config")

    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("loading cached dataset...")
        loaded_dev_activations = torch.load(
            f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(
            f"loaded cached dataset from {loaded_dev_activations['timestamp']}"
        )
    else:
        print("loading dataset...")
        devset = utils.sample_rp1t(tokenizer,
                                   args.devset_size,
                                   args.ctx_size,
                                   nproc=args.sample_proc)
        dev_emb = model.model.embed_tokens(devset)
        after_layer = -1
        print("loaded dataset!")

    print(f"dev_emb dtype: {dev_emb.dtype}")
    print(f"dev_emb shape: {dev_emb.shape}")
    
    # Ensure dev_emb is in shared memory for multi-process access
    dev_emb = dev_emb.share_memory_()

    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)

    # Prepare attention mask with sliding window for Mixtral
    if hasattr(model.config, 'sliding_window') and model.config.sliding_window is not None:
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            dev_emb[0:args.batch_size],
            0,
            sliding_window=model.config.sliding_window)
    else:
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            dev_emb[0:args.batch_size], 0)

    if args.scratch_path is not None:
        os.makedirs(args.scratch_path, exist_ok=True)
        move_q = mp.Queue()
        move_p = mp.Process(target=move_fn,
                            args=(move_q, args.async_copy_speed))
        move_p.start()
    else:
        move_q = None

    # Create a lock for synchronizing dev_emb updates
    manager = mp.Manager()
    dev_emb_lock = manager.Lock()

    for transformer_layer_index in range(len(model.model.layers)):
        if transformer_layer_index <= after_layer:
            print(
                f"skipping layer {transformer_layer_index} because it is before cached activations at layer {after_layer}"
            )
            continue

        transformer_layer = model.model.layers[transformer_layer_index]

        # Verify layer structure
        try:
            verify_layer_structure(transformer_layer, args.num_experts, transformer_layer_index)
        except ValueError as e:
            print(f"Error: {e}")
            raise

        # Calculate number of GPUs to use
        chunk_size = min(args.chunk_size, len(dev_emb))
        
        # Ensure chunk_size is divisible by batch_size
        if chunk_size % args.batch_size != 0:
            chunk_size = (chunk_size // args.batch_size) * args.batch_size
            if chunk_size == 0:
                chunk_size = args.batch_size
        
        total_chunks = (len(dev_emb) + chunk_size - 1) // chunk_size
        ngpus = min(torch.cuda.device_count(), total_chunks)
        
        if ngpus == 0:
            raise RuntimeError("No CUDA devices available")

        print(f"Processing layer {transformer_layer_index} with {ngpus} GPUs, "
              f"chunk_size={chunk_size}, total_chunks={total_chunks}")

        # Create queues
        ctx = mp.get_context('spawn')
        in_q = ctx.Queue()
        out_q = ctx.Queue()
        result_q = ctx.Queue()

        # Start accumulation process
        accumulate_proc = ctx.Process(
            target=accumulate_mixtral,
            args=(out_q, move_q, ngpus, args, transformer_layer_index))
        accumulate_proc.start()

        # Start forward processes
        forward_procs = []
        for i in range(ngpus):
            p = ctx.Process(
                target=forward_layer_mixtral,
                args=(transformer_layer, position_ids, attention_mask,
                      args.batch_size, i, in_q, out_q, result_q, args.num_experts))
            p.start()
            forward_procs.append(p)

        # Distribute chunks to workers
        chunks = distribute_chunks_to_gpus(dev_emb, chunk_size, ngpus)
        total_chunks_count = len(chunks)
        
        for start_idx, end_idx in chunks:
            chunk_data = dev_emb[start_idx:end_idx].clone()
            in_q.put(((start_idx, end_idx), chunk_data))

        # Signal workers to finish
        for i in range(ngpus):
            in_q.put(None)

        # Start result collection process
        collect_proc = ctx.Process(
            target=collect_results,
            args=(result_q, dev_emb, total_chunks_count, dev_emb_lock))
        collect_proc.start()

        # Wait for all processes to complete
        for p in forward_procs:
            p.join()

        collect_proc.join()
        accumulate_proc.join()

        # Cleanup
        transformer_layer.cpu()
        model.model.layers[transformer_layer_index] = None
        utils.clean()

        # Save activations periodically
        if args.save_activations and (
                transformer_layer_index % args.act_save_rate == 0 or
                transformer_layer_index == len(model.model.layers) - 1):
            save_activation_checkpoint(args, dev_emb, transformer_layer_index, move_q)

        print(f"done processing layer {transformer_layer_index}")

    if args.scratch_path is not None:
        move_q.put(None)
        move_p.join()

    print("All layers processed successfully!")


def save_activation_checkpoint(args, dev_emb, transformer_layer_index, move_q):
    """Save activation checkpoint to disk."""
    checkpoint_data = {
        'dev_emb': dev_emb.clone(),  # Clone to avoid issues with shared memory
        'after_layer': transformer_layer_index,
        'timestamp': str(datetime.datetime.now())
    }
    
    if args.scratch_path is not None:
        scratch_path = f'{args.scratch_path}/dev_activations.pt'
        if os.path.exists(scratch_path):
            print('not saving layer since disk is too slow')
        else:
            torch.save(checkpoint_data, scratch_path)
            move_q.put((scratch_path, f'{args.save_path}/dev_activations.pt'))
    else:
        torch.save(checkpoint_data, f'{args.save_path}/dev_activations.pt')


if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Arguments: {args}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    main(args)