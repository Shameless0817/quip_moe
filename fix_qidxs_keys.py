"""
修复已保存模型中的 Qidxs 键名问题
将 Qidxs_0 重命名为 Qidxs
"""
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import os
from pathlib import Path
import json
from tqdm import tqdm

def fix_safetensors_file(input_file, output_file):
    """修复单个 safetensors 文件中的键名"""
    print(f"处理文件: {input_file}")
    
    # 加载原始文件
    state_dict = load_file(input_file)
    
    # 读取原始 metadata
    metadata = {}
    with safe_open(input_file, framework='pt', device='cpu') as f:
        metadata = f.metadata()
    
    # 如果 metadata 为 None，设置默认值
    if metadata is None:
        metadata = {"format": "pt"}
    
    # 重命名键
    new_state_dict = {}
    renamed_count = 0
    
    for key, value in tqdm(state_dict.items(), desc="重命名键"):
        # 如果键名包含 Qidxs_0，改为 Qidxs
        if '.Qidxs_0' in key:
            new_key = key.replace('.Qidxs_0', '.Qidxs')
            new_state_dict[new_key] = value
            renamed_count += 1
        # 如果有 Qidxs_1, Qidxs_2 等，跳过（我们只需要 Qidxs_0）
        elif '.Qidxs_' in key and not key.endswith('.Qidxs_0'):
            # 跳过其他分片
            continue
        else:
            new_state_dict[key] = value
    
    print(f"  重命名了 {renamed_count} 个 Qidxs 键")
    
    # 保存到新文件，保留 metadata
    save_file(new_state_dict, output_file, metadata=metadata)
    print(f"  已保存到: {output_file}")
    
    return renamed_count

def fix_model_directory(model_dir):
    """修复整个模型目录"""
    model_dir = Path(model_dir)
    output_dir = model_dir.parent / (model_dir.name + "_fixed")
    output_dir.mkdir(exist_ok=True)
    
    print(f"输入目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    # 找到所有 safetensors 文件
    safetensors_files = list(model_dir.glob("*.safetensors"))
    
    if not safetensors_files:
        print("错误: 没有找到 safetensors 文件")
        return
    
    # 处理每个文件
    total_renamed = 0
    for st_file in safetensors_files:
        if st_file.name == "model.safetensors.index.json":
            continue
        
        output_file = output_dir / st_file.name
        renamed = fix_safetensors_file(str(st_file), str(output_file))
        total_renamed += renamed
    
    # 复制其他文件（config.json 等）
    print("\n复制配置文件...")
    for file in model_dir.glob("*.json"):
        if not file.name.endswith(".safetensors.index.json"):
            output_file = output_dir / file.name
            import shutil
            shutil.copy(file, output_file)
            print(f"  复制: {file.name}")
    
    # 如果有 index.json，需要更新它（但键名映射保持不变，因为已经重命名了）
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # 更新 weight_map 中的键名
        new_weight_map = {}
        for key, file_name in index_data['weight_map'].items():
            if '.Qidxs_0' in key:
                new_key = key.replace('.Qidxs_0', '.Qidxs')
                new_weight_map[new_key] = file_name
            elif '.Qidxs_' in key and not key.endswith('.Qidxs_0'):
                # 跳过其他分片
                continue
            else:
                new_weight_map[key] = file_name
        
        index_data['weight_map'] = new_weight_map
        
        output_index = output_dir / "model.safetensors.index.json"
        with open(output_index, 'w') as f:
            json.dump(index_data, f, indent=2)
        print(f"  更新: model.safetensors.index.json")
    
    print("\n" + "="*80)
    print(f"完成！总共重命名了 {total_renamed} 个 Qidxs 键")
    print(f"修复后的模型已保存到: {output_dir}")
    print("\n使用修复后的模型:")
    print(f"  python reload_model_lora.py --model_dir {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="修复 Qidxs 键名")
    parser.add_argument("model_dir", type=str, help="模型目录路径")
    
    args = parser.parse_args()
    
    fix_model_directory(args.model_dir)
