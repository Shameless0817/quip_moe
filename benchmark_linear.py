import torch
import os
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    torch_dtype='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='cpu'  # 先加载到CPU
)
model.eval()
print(model)