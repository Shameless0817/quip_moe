"""
简单示例：加载并使用HuggingFace上的2bit量化模型
用法：python run_2bit_model.py --hf_path relaxml/Llama-2-7b-E8P-2Bit
"""
import argparse
import torch
from lib.utils.unsafe_import import model_from_hf_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_path', default='relaxml/Llama-2-7b-E8P-2Bit', 
                       type=str, help='HuggingFace模型路径')
    parser.add_argument('--prompt', default='Once upon a time', 
                       type=str, help='输入提示词')
    parser.add_argument('--max_length', default=100, type=int, 
                       help='生成的最大长度')
    parser.add_argument('--no_use_cuda_graph', action='store_true')
    parser.add_argument('--no_use_flash_attn', action='store_true')
    args = parser.parse_args()

    print(f"正在加载模型: {args.hf_path}")
    print("这可能需要几分钟时间...")
    
    # 加载模型
    model, model_str = model_from_hf_path(
        args.hf_path,
        use_cuda_graph=not args.no_use_cuda_graph,
        use_flash_attn=not args.no_use_flash_attn
    )
    
    print(f"✓ 模型加载成功: {model_str}")
    print(f"输入提示词: {args.prompt}")
    print("-" * 50)
    

    print("\n提示：")
    print("1. 评估困惑度: python eval/eval_ppl.py --hf_path", args.hf_path)
    print("2. Zero-shot评估: python eval/eval_zeroshot.py --hf_path", args.hf_path)
    print("3. 速度测试: python eval/eval_speed.py --hf_path", args.hf_path)

if __name__ == '__main__':
    main()
