import argparse
import json
import time

import glog
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', default='meta-llama/Llama-2-7b-hf', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--prefill_sizes', default='32,64,128,256,512,1024,2048', type=str,
                    help='Comma-separated list of prefill sizes to test')
parser.add_argument('--decode_steps', default=128, type=int,
                    help='Number of decode steps to generate after prefill')
parser.add_argument('--warmup_samples', default=5, type=int,
                    help='Number of warmup iterations')
parser.add_argument('--test_samples', default=20, type=int,
                    help='Number of test iterations per prefill size')
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')
parser.add_argument('--output_json', default=None, type=str,
                    help='Path to save results as JSON')


def benchmark_prefill(model, input_ids, warmup_samples=5, test_samples=20):
    """
    Benchmark prefill phase (first forward pass with full context)
    """
    # Warmup
    for _ in range(warmup_samples):
        _ = model(input_ids, use_cache=False)
        torch.cuda.synchronize()
    
    # Actual test
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(test_samples):
        _ = model(input_ids, use_cache=False)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_latency = (end - start) / test_samples
    return avg_latency


def benchmark_decode(model, past_key_values, warmup_samples=5, test_samples=20):
    """
    Benchmark decode phase (incremental generation with KV cache)
    """
    batch_size = past_key_values[0][0].shape[0]
    single_token = torch.ones(batch_size, 1, dtype=torch.long).cuda()
    
    # Warmup
    for _ in range(warmup_samples):
        _ = model(single_token, past_key_values=past_key_values, use_cache=True)
        torch.cuda.synchronize()
    
    # Actual test
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(test_samples):
        _ = model(single_token, past_key_values=past_key_values, use_cache=True)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_latency = (end - start) / test_samples
    return avg_latency


def benchmark_full_generation(model, input_ids, decode_steps=128):
    """
    Benchmark full generation: prefill + multiple decode steps
    """
    torch.cuda.synchronize()
    start = time.time()
    
    # Prefill phase
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    
    # Decode phase
    current_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    for _ in range(decode_steps - 1):
        outputs = model(current_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        current_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = end - start
    total_tokens = input_ids.shape[1] + decode_steps
    tokens_per_sec = total_tokens / total_time
    
    return total_time, tokens_per_sec


def main(args):
    glog.info(f"Loading model from {args.hf_path}")
    model, model_str = model_from_hf_path(
        args.hf_path,
        use_cuda_graph=not args.no_use_cuda_graph,
        use_flash_attn=not args.no_use_flash_attn)
    
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    
    # Parse prefill sizes
    prefill_sizes = [int(x) for x in args.prefill_sizes.split(',')]
    glog.info(f"Testing prefill sizes: {prefill_sizes}")
    
    results = []
    
    for prefill_size in tqdm(prefill_sizes, desc="Testing prefill sizes"):
        glog.info(f"\n{'='*60}")
        glog.info(f"Testing prefill size: {prefill_size}")
        glog.info(f"{'='*60}")
        
        # Create input tokens
        input_ids = torch.randint(
            0, tokenizer.vocab_size, 
            (args.batch_size, prefill_size), 
            dtype=torch.long
        ).cuda()
        
        # Reset model if supported
        if hasattr(model, 'reset') and not args.no_use_cuda_graph:
            model.reset()
        
        # 1. Benchmark prefill phase only
        glog.info("Benchmarking prefill phase...")
        prefill_latency = benchmark_prefill(
            model, input_ids, 
            warmup_samples=args.warmup_samples,
            test_samples=args.test_samples
        )
        prefill_tokens_per_sec = prefill_size / prefill_latency
        
        # 2. Benchmark decode phase only
        glog.info("Benchmarking decode phase...")
        # First do a prefill to get KV cache
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
        
        decode_latency = benchmark_decode(
            model, past_key_values,
            warmup_samples=args.warmup_samples,
            test_samples=args.test_samples
        )
        decode_tokens_per_sec = 1.0 / decode_latency
        
        # 3. Benchmark full generation (prefill + decode)
        glog.info(f"Benchmarking full generation ({args.decode_steps} decode steps)...")
        total_time, total_tokens_per_sec = benchmark_full_generation(
            model, input_ids, decode_steps=args.decode_steps
        )
        
        result = {
            'prefill_size': prefill_size,
            'batch_size': args.batch_size,
            'decode_steps': args.decode_steps,
            'prefill': {
                'latency_ms': prefill_latency * 1000,
                'tokens_per_sec': prefill_tokens_per_sec,
            },
            'decode': {
                'latency_ms': decode_latency * 1000,
                'tokens_per_sec': decode_tokens_per_sec,
            },
            'full_generation': {
                'total_time_s': total_time,
                'total_tokens': prefill_size + args.decode_steps,
                'tokens_per_sec': total_tokens_per_sec,
            }
        }
        results.append(result)
        
        # Print results
        glog.info(f"\nResults for prefill_size={prefill_size}:")
        glog.info(f"  Prefill:")
        glog.info(f"    Latency: {prefill_latency*1000:.2f} ms")
        glog.info(f"    Throughput: {prefill_tokens_per_sec:.2f} tokens/s")
        glog.info(f"  Decode:")
        glog.info(f"    Latency: {decode_latency*1000:.2f} ms/token")
        glog.info(f"    Throughput: {decode_tokens_per_sec:.2f} tokens/s")
        glog.info(f"  Full Generation (prefill + {args.decode_steps} decode):")
        glog.info(f"    Total time: {total_time:.2f} s")
        glog.info(f"    Average throughput: {total_tokens_per_sec:.2f} tokens/s")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Prefill Size':<15} {'Prefill (ms)':<15} {'Prefill (tok/s)':<18} "
          f"{'Decode (ms)':<15} {'Decode (tok/s)':<18} {'Full Gen (tok/s)':<18}")
    print("-"*100)
    for r in results:
        print(f"{r['prefill_size']:<15} "
              f"{r['prefill']['latency_ms']:<15.2f} "
              f"{r['prefill']['tokens_per_sec']:<18.2f} "
              f"{r['decode']['latency_ms']:<15.2f} "
              f"{r['decode']['tokens_per_sec']:<18.2f} "
              f"{r['full_generation']['tokens_per_sec']:<18.2f}")
    print("="*100)
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({
                'model': args.hf_path,
                'batch_size': args.batch_size,
                'decode_steps': args.decode_steps,
                'use_cuda_graph': not args.no_use_cuda_graph,
                'use_flash_attn': not args.no_use_flash_attn,
                'results': results
            }, f, indent=2)
        glog.info(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    main(args)
