import argparse
import json
import math
import os
import random

import datasets
import glog
import torch
from tqdm import tqdm

from lib.utils import gptq_data_utils
from lib.utils.unsafe_import import model_from_hf_path
from transformers import AutoTokenizer

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--seqlen', default=4096, type=int)
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    datasets = ['wikitext2', 'c4']
    glog.info(f'Evaluating model from hf_path={args.hf_path}')
    model, model_str = model_from_hf_path(
        args.hf_path,
        use_cuda_graph=not args.no_use_cuda_graph,
        use_flash_attn=not args.no_use_flash_attn)
    print(model)
    # input_texts = "The capital of France is"
    # tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    # input_ids = tokenizer(input_texts, return_tensors='pt').input_ids.cuda() 
    # output_texts = model.generate(input_ids, max_length=50)
    # print("Generated text:", tokenizer.batch_decode(output_texts, skip_special_tokens=True))
    # # exit() 


    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                    seed=args.seed,
                                                    seqlen=args.seqlen,
                                                    model=model_str)
        nsamples = input_tok.numel() // args.seqlen
        input_tok = input_tok[0, :(args.seqlen * nsamples)].view(
            nsamples, args.seqlen)

        if not args.no_use_cuda_graph:
            # 只在模型有 reset 方法时调用（某些模型可能不支持）
            if hasattr(model, 'reset'):
                model.reset()

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input,
                           use_cache=False,
                           output_hidden_states=False,
                           output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        glog.info(f'{dataset} perplexity: {ppl}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
