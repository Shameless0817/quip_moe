#!/usr/bin/env python3
"""
测试 hfize_mixtral.py 的模型加载和设备分配
检查是否解决了 "Expected all tensors to be on the same device" 错误
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', default="/fact_home/zeyuli/quip_sharp/quantized_mixtral_noft", type=str)
parser.add_argument('--hf_output_path', default="./mixtral_8x7b_quip", type=str)
args = parser.parse_args()

print("="*80)
print("测试 hfize_mixtral.py 模型加载")
print("="*80)

# 执行主函数
from hfize_mixtral import main
try:
    print("\n开始加载模型...")
    main(args)
    print("\n✓ 模型加载成功！")
    
    # 如果成功创建了模型文件，尝试测试推理
    if os.path.exists(args.hf_output_path):
        print("\n" + "="*80)
        print("测试模型推理")
        print("="*80)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("\n加载保存的模型...")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_output_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_output_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("模型加载成功！")
        print(f"模型类型: {type(model)}")
        
        # 检查设备分配
        print("\n检查设备分配:")
        for name, param in model.named_parameters():
            if 'self_attn' in name and any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                print(f"  {name}: {param.device}")
                break
        
        # 测试简单推理
        print("\n测试文本生成...")
        text = "Hello, I am a"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print(f"输入设备: {inputs['input_ids'].device}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n生成结果: {generated_text}")
        print("\n✓ 推理测试成功！设备问题已解决！")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("测试完成")
print("="*80)
