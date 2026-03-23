#!/usr/bin/env bash
# HF_PATH="${HF_PATH:-/fact_home/zeyuli/quip_sharp/deepseek_quip_full_noft}"
HF_PATH="deepseek-ai/deepseek-moe-16b-base"
SEQLEN="${SEQLEN:-4096}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1,2,3}" python -m eval.eval_ppl --hf_path "${HF_PATH}" --seqlen "${SEQLEN}"