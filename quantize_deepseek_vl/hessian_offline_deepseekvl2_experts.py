import argparse
import datetime
import os
import random
import sys

# Add parent directory to path to allow importing lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from lib import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def forward_layer_deepseek(
    layer,
    position_ids,
    attention_mask,
    bs,
    device,
    dev_emb,
    num_experts
):
    """Forward pass for DeepSeek layer - collecting Hessians for Dense or MoE layers."""
    layer = layer.to(device)
    
    # 注意：多模态模型的 position_ids 可能是 2D/3D 的，attention_mask 可能是 4D 的
    # 这里我们不再强制切片，而是根据截获的真实形状进行批次划分
    
    done_hooks = {}
    is_moe = hasattr(layer.mlp, 'experts')

    if is_moe:
        print(f"Collecting Hessian for MoE layer: gate, shared_experts, and {num_experts} routed experts")
        if hasattr(layer.mlp, 'gate'):
            done_hooks['gate'] = utils.register_H_hook(layer.mlp.gate, device)
        if hasattr(layer.mlp, 'shared_experts') and layer.mlp.shared_experts is not None:
            done_hooks['shared_gate_proj'] = utils.register_H_hook(layer.mlp.shared_experts.gate_proj, device)
            done_hooks['shared_up_proj'] = utils.register_H_hook(layer.mlp.shared_experts.up_proj, device)
            done_hooks['shared_down_proj'] = utils.register_H_hook(layer.mlp.shared_experts.down_proj, device)
        for expert_idx in range(num_experts):
            expert = layer.mlp.experts[expert_idx]
            done_hooks[f'expert{expert_idx}_gate_proj'] = utils.register_H_hook(expert.gate_proj, device)
            done_hooks[f'expert{expert_idx}_up_proj'] = utils.register_H_hook(expert.up_proj, device)
            done_hooks[f'expert{expert_idx}_down_proj'] = utils.register_H_hook(expert.down_proj, device)
    else:
        print("Collecting Hessian for Dense layer")
        done_hooks['dense_gate_proj'] = utils.register_H_hook(layer.mlp.gate_proj, device)
        done_hooks['dense_up_proj'] = utils.register_H_hook(layer.mlp.up_proj, device)
        done_hooks['dense_down_proj'] = utils.register_H_hook(layer.mlp.down_proj, device)

    # Forward pass through layer
    num_samples = len(dev_emb)
    assert num_samples % bs == 0, f"Number of samples ({num_samples}) must be divisible by batch size ({bs})"
    
    for i in tqdm(range(num_samples // bs), desc="Forward pass"):
        batch = dev_emb[i * bs:(i + 1) * bs].to(device)
        
        # 提取对应的 position_ids 和 attention_mask
        batch_pos_ids = position_ids[i * bs:(i + 1) * bs].to(device) if position_ids is not None else None
        batch_attn_mask = attention_mask[i * bs:(i + 1) * bs].to(device) if attention_mask is not None else None
        
        with torch.no_grad():
            output = layer(
                batch,
                position_ids=batch_pos_ids,
                attention_mask=batch_attn_mask,
                use_cache=False,
                output_attentions=False
            )[0]
        dev_emb[i * bs:(i + 1) * bs].copy_(output.cpu())

    layer = layer.cpu()
    
    results = {}
    for key, done_fn in done_hooks.items():
        results[key] = done_fn()

    return results


def save_hessians(results, args, transformer_layer_index):
    """Process and save Hessians for a single layer."""
    for key, (H, mu, ct) in results.items():
        mu = mu / ct
        H = H / ct
        H.sub_(mu.unsqueeze(-1) @ mu.unsqueeze(0))
        H = (H + H.T) / 2
        
        try:
            eigvals = torch.linalg.eigvalsh(H)
            min_eigval = eigvals.min().item()
            if min_eigval < 0:
                H.add_(torch.eye(H.shape[0], dtype=H.dtype, device=H.device) * abs(min_eigval) * 1.1)
        except:
            pass

        save_path = f"{args.save_path}/{transformer_layer_index}_{key}.pt"
        torch.save({
            'flatH': utils.sym_to_flat(H.to(torch.float32)),
            'mu': mu.to(torch.float32),
            'n': H.shape[0],
            'ct': ct
        }, save_path)
        print(f"Saved {save_path}")
    del results


def count_linear_layers_deepseek(layer, num_experts):
    """Count expected linear layers in DeepSeek layer."""
    if hasattr(layer.mlp, 'experts'):
        count = 1 + (num_experts * 3)
        if hasattr(layer.mlp, 'shared_experts') and layer.mlp.shared_experts is not None:
            count += 3
        return count
    else:
        return 3


def get_multimodal_calibration_data(
    processor, 
    batch_size, 
    num_batches
):
    """Generate multimodal calibration data for Deepseek-VL2."""
    from datasets import load_dataset
    print("Loading dataset for multimodal calibration...")
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True)
    dataset_iter = iter(dataset)

    for _ in range(batch_size):
        sample = next(dataset_iter)

        image = sample["image"].convert("RGB")
        batch_images.append(image)

        question = sample["question"]

        answer = sample.get("multiple_choice_answer", "")
        if not answer and "answers" in sample and len(sample["answers"]) > 0:
            answer = sample["answers"][0]
    
        # 构造Deepseek-VL2要求的格式

        conversations = [
            {
                "role": "User",
                "content": f"<image>\n{question}"
            },
            {
                "role": "Assistant",
                "content": answer
            }
        ]
        batch_conversations.append(conversations)
        
        if not batch_conversations:
            raise ValueError("No valid samples found in dataset for calibration. Please check the dataset and try again.")
            break

        inputs = processor(
            conversations=batch_conversations,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )
        inputs_list.append(inputs)

        print(f"Prepared batch {batch_idx + 1}/{num_batches} for calibration")
        
    return inputs_list


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading processor and model...")
    from deepseek_vl2.models import DeepSeekVLV2ForCausalLM
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = DeepSeekVLV2ForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("Loaded model!")

    # 定位 VL2 内部的语言模型层
    # DeepSeek-VL2 通常将 LLM 封装在 language_model 属性下
    if hasattr(model, 'language_model'):
        lm_layers = model.language_model.model.layers
        lm_config = model.language_model.config
    else:
        lm_layers = model.model.layers
        lm_config = model.config

    if hasattr(lm_config, 'n_routed_experts'):
        args.num_experts = lm_config.n_routed_experts
        print(f"Detected {args.num_experts} routed experts from model config")

    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("Loading cached dataset...")
        loaded_dev_activations = torch.load(f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        # 兼容旧版本缓存文件
        position_ids = loaded_dev_activations.get('position_ids', None)
        attention_mask = loaded_dev_activations.get('attention_mask', None)
        print(f"Loaded cached dataset from {loaded_dev_activations.get('timestamp', 'unknown time')}")
    else:
        print("Capturing initial embeddings from the first LM layer...")
        after_layer = -1
        
        # 获取多模态校准数据
        num_batches = args.devset_size // args.batch_size
        calib_batches = get_multimodal_calibration_data(processor, args.batch_size, num_batches)
        
        captured_inputs = {'hidden_states': [], 'position_ids': [], 'attention_mask': []}
        
        # 定义拦截器（Hook）挂载在语言模型的第一层
        def capture_hook(module, args, kwargs):
            captured_inputs['hidden_states'].append(args[0].detach().cpu())
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                captured_inputs['position_ids'].append(kwargs['position_ids'].detach().cpu())
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                captured_inputs['attention_mask'].append(kwargs['attention_mask'].detach().cpu())
            raise RuntimeError("Captured initial inputs, stopping forward pass.")

        hook_handle = lm_layers[0].register_forward_pre_hook(capture_hook, with_kwargs=True)
        
        model = model.to(device)
        model.eval()
        
        for batch_inputs in tqdm(calib_batches, desc="Capturing embeddings"):
            batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
            try:
                with torch.no_grad():
                    model(**batch_inputs)
            except RuntimeError as e:
                if "Captured initial inputs" not in str(e):
                    raise e
                    
        hook_handle.remove()
        model = model.cpu()
        torch.cuda.empty_cache()
        
        # 拼接所有批次的数据
        dev_emb = torch.cat(captured_inputs['hidden_states'], dim=0)
        position_ids = torch.cat(captured_inputs['position_ids'], dim=0) if captured_inputs['position_ids'] else None
        attention_mask = torch.cat(captured_inputs['attention_mask'], dim=0) if captured_inputs['attention_mask'] else None
        
        print(f"Captured dev_emb shape: {dev_emb.shape}")
        if position_ids is not None: print(f"Captured position_ids shape: {position_ids.shape}")
        if attention_mask is not None: print(f"Captured attention_mask shape: {attention_mask.shape}")

    dev_emb.requires_grad_(False)

    # Process each layer
    num_layers = len(lm_layers)
    for transformer_layer_index in range(num_layers):
        if transformer_layer_index <= after_layer:
            print(f"Skipping layer {transformer_layer_index}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing layer {transformer_layer_index}/{num_layers - 1}")
        print(f"{'='*50}")

        transformer_layer = lm_layers[transformer_layer_index]

        linear_count = len([m for m in transformer_layer.modules() if isinstance(m, torch.nn.Linear)])
        expected_count = count_linear_layers_deepseek(transformer_layer, args.num_experts)
        print(f"Found {linear_count} linear layers, collecting Hessians for {expected_count} target layers")

        results = forward_layer_deepseek(
            transformer_layer,
            position_ids,
            attention_mask,
            args.batch_size,
            device,
            dev_emb,
            args.num_experts
        )

        save_hessians(results, args, transformer_layer_index)

        if args.save_activations:
            torch.save({
                'dev_emb': dev_emb,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'after_layer': transformer_layer_index,
                'timestamp': str(datetime.datetime.now())
            }, f"{args.save_path}/dev_activations.pt")

        torch.cuda.empty_cache()

    print("\nAll layers processed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect Hessian matrices for DeepSeek-VL2 LM layers')
    parser.add_argument('--base_model', type=str, required=True, help='HuggingFace model name or path')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save Hessian matrices')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for forward pass (keep small for VLMs)')
    parser.add_argument('--devset_size', type=int, default=128, help='Number of samples for calibration')
    parser.add_argument('--num_experts', type=int, default=64, help='Number of routed experts')
    parser.add_argument('--save_activations', action='store_true', help='Save intermediate activations')
    
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)