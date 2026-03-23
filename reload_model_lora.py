import torch
import torch.nn as nn
import json
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from safetensors.torch import load_file  # 核心：用于加载 safetensors
from lib.utils import gptq_data_utils
# from lib.utils.unsafe_import import model_from_hf_path
from lib.utils.unsafe_import import lora_model_from_hf_path
from lib.linear.quantized_linear_withlora import QuantizedLinearWithlora

# 引入你的自定义库
# 假设你的库路径是 lib.codebook.latticee8_padded12
# from lib.codebook.latticee8_padded12 import QuantizedE8P12LinearWithLoRA, E8P12_codebook

from lib.linear.quantized_linear_withlora import QuantizedLinearWithlora

def replace_linear_with_custom(
    model, 
    lora_rank, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]):
    """
    遍历模型，将指定的 Linear 层替换为 QuantizedE8P12LinearWithLoRA 占位符
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_with_custom(module, lora_rank, target_modules)
        
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            in_features = module.in_features
            out_features = module.out_features
            device = module.weight.device
            
            class MockQuantizedWrapper:
                def __init__(self, d_in, d_out, dev):
                    self.codebook = E8P12_codebook(inference=True).to(torch.float16).to(dev)
                    self.scale = 32.0 
                    self.in_features = d_in
                    self.out_features = d_out
                    self.weight = torch.zeros(1, device=dev)
            
            # 初始化自定义层
            new_layer = QuantizedE8P12LinearWithLoRA(
                original_module=MockQuantizedWrapper(in_features, out_features, device),
                lora_rank=lora_rank,
                device=device
            )
            
            setattr(model, name, new_layer)

def load_weights_from_hf_format(model, model_dir):
    """
    智能加载权重：支持 safetensors, bin, 以及分片(sharded)模型
    """
    print(f"开始加载权重: {model_dir}")
    
    # 1. 检查是否有索引文件 (Sharded Model)
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    safe_tensors_files = []
    
    if os.path.exists(index_file):
        print("检测到分片模型 (Sharded Safetensors)...")
        with open(index_file, "r") as f:
            index_data = json.load(f)
        # 获取所有唯一的权重文件名
        weight_filenames = set(index_data["weight_map"].values())
        safe_tensors_files = [os.path.join(model_dir, f) for f in weight_filenames]
        safe_tensors_files.sort() # 排序以保证顺序（虽然不严格要求）
    else:
        # 2. 如果没有索引，查找单个 safetensors 文件
        possible_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
        if len(possible_files) > 0:
            print("检测到单文件 Safetensors...")
            safe_tensors_files = possible_files
        else:
            # 3. 回退到 pytorch_model.bin (为了兼容性)
            bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
            if os.path.exists(bin_index):
                print("检测到分片 PyTorch Bin...")
                with open(bin_index, "r") as f:
                    index_data = json.load(f)
                weight_filenames = set(index_data["weight_map"].values())
                # 这里返回的是 bin 文件列表，后面加载逻辑需要区分
                bin_files = [os.path.join(model_dir, f) for f in weight_filenames]
                for bf in tqdm(bin_files, desc="Loading Bin Shards"):
                    state_dict = torch.load(bf, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                return
            else:
                bin_file = os.path.join(model_dir, "pytorch_model.bin")
                if os.path.exists(bin_file):
                    print("检测到单文件 PyTorch Bin...")
                    state_dict = torch.load(bin_file, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                    return
                else:
                    raise FileNotFoundError(f"在 {model_dir} 中未找到 .safetensors 或 .bin 权重文件")

    # 4. 执行 Safetensors 加载 (核心逻辑)
    for st_file in tqdm(safe_tensors_files, desc="Loading Safetensors Shards"):
        # load_file 是 safetensors 库提供的直接读取方法
        state_dict = load_file(st_file, device="cpu")
        
        # 处理可能的键名前缀问题 (如编译模型时产生的 _orig_mod)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 增量加载：strict=False 允许我们一次只加载一部分权重
        # PyTorch 会自动匹配键名，忽略不在当前 state_dict 中的键
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        # 释放内存
        del state_dict
        del new_state_dict
        torch.cuda.empty_cache()

    print("所有权重分片加载完毕。")

def load_reconstructed_model(model_dir, device="cuda"):
    """
    主加载流程
    """
    # 1. 读取配置
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 尝试从 config.json 或 lora_config.json 中获取 rank
    # 如果你保存时没有把 lora_rank 写入 config.json，这里需要手动指定
    lora_rank = config.get('lora_rank', 16) 
    # 或者尝试读取 lora_config.json
    lora_config_path = os.path.join(model_dir, "lora_config.json")
    if os.path.exists(lora_config_path):
        with open(lora_config_path, 'r') as f:
            lora_conf = json.load(f)
            lora_rank = lora_conf.get('lora_rank', lora_rank)

    original_model_path = config.get('_name_or_path', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
    
    print(f"基础模型路径: {original_model_path}")
    print(f"LoRA Rank: {lora_rank}")

    # 2. 加载骨架 (CPU)
    print("加载模型骨架...")
    model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float16,
        device_map="cpu", 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 3. 替换层结构
    print("替换网络层结构...")
    target_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
    replace_linear_with_custom(model, lora_rank, target_layers)

    # 4. 加载权重 (支持 Safetensors)
    load_weights_from_hf_format(model, model_dir)
    
    # 5. 移动到 GPU
    print(f"移动模型到 {device}...")
    model.to(device)
    model.eval()
    
    return model

def eval_perplexity(model, tokenizer, dataset_name="wikitext", subset="wikitext-2-raw-v1", device="cuda"):
    """
    计算 PPL
    """
    print(f"\n开始计算 PPL ({dataset_name})...")
    test = load_dataset(dataset_name, subset, split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    pbar = tqdm(range(0, seq_len, stride))
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if len(nlls) > 0:
            pbar.set_description(f"Current PPL: {torch.exp(torch.stack(nlls).mean()):.2f}")
            
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nFinal Perplexity: {ppl:.4f}")
    return ppl

if __name__ == "__main__":
    # 配置
    MODEL_DIR = "/fact_home/zeyuli/quip_sharp/mixtral_lora_converted_hf_v2" # 修复后的模型路径
    # MODEL_DIR = "/fact_home/zeyuli/quip_sharp/mixtral_lora_converted_hf_v2" # 原始的模型路径
    # MODEL_DIR = "/fact_data/zeyuli/mixtral_8x7b_quip_full_noft"
    TOKENIZER_PATH = "mistralai/Mixtral-8x7B-v0.1" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # 加载模型（注意：model_from_hf_path 返回 (model, model_str) 的 tuple）
    # model = load_reconstructed_model(MODEL_DIR, device)
    model, model_str = lora_model_from_hf_path(MODEL_DIR, use_cuda_graph=False, use_flash_attn=False)
    print(f"Loaded model: {model_str}")
    print(model)
    # 测试生成
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(out[0]))
    exit() 
    
    # 计算 PPL
    eval_perplexity(model, tokenizer, device=device)