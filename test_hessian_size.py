import os
import torch
import argparse

def test_hessian_sizes(hessian_dir, layers=[0, 1]):
    """
    打印指定层的Hessian矩阵文件中存储的参数size
    
    Args:
        hessian_dir: Hessian矩阵文件所在目录
        layers: 要检查的层索引列表
    """
    print(f"检查目录: {hessian_dir}")
    print("="*80)
    
    for layer_idx in layers:
        print(f"\n{'='*80}")
        print(f"Layer {layer_idx}")
        print(f"{'='*80}")
        
        # 查找该层的所有文件
        layer_files = [f for f in os.listdir(hessian_dir) if f.startswith(f"{layer_idx}_")]
        layer_files.sort()
        
        if not layer_files:
            print(f"  未找到第 {layer_idx} 层的文件")
            continue
        
        for filename in layer_files:
            filepath = os.path.join(hessian_dir, filename)
            print(f"\n文件: {filename}")
            
            try:
                # 加载文件
                data = torch.load(filepath, map_location='cpu')
                
                # 打印所有键和对应的size
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key:15s}: shape={list(value.shape)}, dtype={value.dtype}, numel={value.numel()}")
                    else:
                        print(f"  {key:15s}: {type(value).__name__} = {value}")
                
                # 如果有flatH，计算完整的H矩阵维度
                if 'flatH' in data and 'n' in data:
                    n = data['n']
                    flatH_len = data['flatH'].numel()
                    expected_len = n * (n + 1) // 2
                    print(f"  {'完整H矩阵':15s}: 应该是 {n}x{n} (flatH长度: {flatH_len}, 期望: {expected_len})")
                    if flatH_len != expected_len:
                        print(f"  ⚠️  警告: flatH长度不匹配!")
                        
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
        
    print(f"\n{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试Hessian矩阵文件的参数大小')
    parser.add_argument('--hessian_dir', type=str, required=True,
                       help='Hessian矩阵文件所在目录')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 1],
                       help='要检查的层索引 (默认: 0 1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hessian_dir):
        print(f"错误: 目录不存在: {args.hessian_dir}")
        exit(1)
    
    test_hessian_sizes(args.hessian_dir, args.layers)