import argparse
import os
import time

import glog
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer

from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', default='meta-llama/Llama-2-70b-hf', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--prefill_seqlen', default=128, type=int)  # 修改：prefill的序列长度
parser.add_argument('--decode_steps', default=50, type=int)    # 修改：decode生成的token数
parser.add_argument('--samples', default=50, type=int)
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    model, model_str = model_from_hf_path(
        args.hf_path,
        use_cuda_graph=not args.no_use_cuda_graph,
        use_flash_attn=not args.no_use_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    
    prefill_token = inputs['input_ids'][0:1, 0:1].cuda().repeat(
        args.batch_size, args.prefill_seqlen)
    
    print("Warming up with prefill...")
    outputs = model(prefill_token, use_cache=True)
    past_key_values = outputs.past_key_values  # 保存KV cache
    
    decode_token = inputs['input_ids'][0:1, 0:1].cuda().repeat(args.batch_size, 1)
    
    _ = model(decode_token, past_key_values=past_key_values, use_cache=True)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(args.samples):
        outputs = model(decode_token, past_key_values=past_key_values, use_cache=True)
        # 注意：这里为了纯粹测试decode性能，重复使用相同的past_key_values
        # 实际生成中，past_key_values会不断增长
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time_sec = (end - start) / args.samples
    avg_time_ms = avg_time_sec * 1000
    
    print(f'Average DECODE time per token: {avg_time_sec:.6f} seconds ({avg_time_ms:.3f} ms)')
    print(f'Throughput: {1/avg_time_sec:.2f} tokens/second')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)