import torch
import os

def inspect_pt_files(folder_path, num_files=5):
    """
    检查文件夹中前 num_files 个 .pt 文件的内容形状
    """
    # 获取所有 .pt 文件并按名称排序（可选）
    pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    pt_files.sort()  # 保证顺序一致
    pt_files = pt_files[:num_files]

    for file_name in pt_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"\n=== {file_name} ===")
        try:
            # 加载到 CPU，避免 GPU 内存占用
            data = torch.load(file_path, map_location='cpu')

            # 递归打印张量形状
            def print_shapes(obj, indent=0):
                prefix = " " * indent
                if torch.is_tensor(obj):
                    print(f"{prefix}Tensor shape: {tuple(obj.shape)}")
                    print(f"{prefix}Tensor dtype: {obj.dtype}")
                elif isinstance(obj, dict):
                    print(f"{prefix}Dict with keys:")
                    for k, v in obj.items():
                        print(f"{prefix}  {k}: ", end="")
                        print_shapes(v, indent + 2)
                elif isinstance(obj, (list, tuple)):
                    print(f"{prefix}{type(obj).__name__} of length {len(obj)}")
                    for i, item in enumerate(obj):
                        print(f"{prefix}  [{i}]: ", end="")
                        print_shapes(item, indent + 2)
                else:
                    print(f"{prefix}{type(obj)} (not a tensor/container)")

            print_shapes(data)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

if __name__ == "__main__":
    folder = input("请输入文件夹路径: ").strip()
    inspect_pt_files(folder)