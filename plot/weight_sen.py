import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc

# ==========================================
# 1. Configuration
# ==========================================
MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
# If you don't have the full model downloaded or enough VRAM, 
# you can test this logic with a smaller MoE model like "openlm-research/open_llama_3b_v2" 
# or by creating a dummy config. For this script, I will assume real loading.

# Define the quantization settings you want to compare
# Format: (bits, group_size, is_symmetric)
# group_size -1 means per-channel (per output channel usually)
QUANT_CONFIGS = [
    {"name": "W8_Channel_Sym", "bits": 8, "group_size": -1, "sym": True},
    {"name": "W4_Channel_Sym", "bits": 4, "group_size": -1, "sym": True},
    {"name": "W4_G128_Asym",   "bits": 4, "group_size": 128, "sym": False},
    {"name": "W3_G128_Asym",   "bits": 3, "group_size": 128, "sym": False},
    {"name": "W2_G128_Asym",   "bits": 2, "group_size": 128, "sym": False},
    {"name": "W2_G128_Sym",   "bits": 2, "group_size": 128, "sym": True},
]

# ==========================================
# 2. Quantization Simulation Functions
# ==========================================
def quantize_weight(w, bits, group_size, sym=True):
    """
    Simulates fake quantization to calculate loss.
    w: Input tensor (Output_Dim, Input_Dim)
    """
    w_orig = w.clone()
    
    # Reshape for grouping
    if group_size > 0:
        orig_shape = w.shape
        # Pad if necessary
        if w.numel() % group_size != 0:
            pad_len = group_size - (w.numel() % group_size)
            w = torch.nn.functional.pad(w.flatten(), (0, pad_len)).reshape(-1, group_size)
        else:
            w = w.reshape(-1, group_size)
    else:
        # Per-channel (usually per output row for Linear layers)
        # Shape becomes (Rows, 1) for scales
        pass 

    # Calculate Scales and Zero-points
    if sym:
        # Symmetric: Range is [-max_abs, max_abs]
        if group_size > 0:
            max_val = w.abs().amax(dim=1, keepdim=True)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            
        max_val = max_val.clamp(min=1e-5)
        q_max = 2**(bits-1) - 1
        scale = max_val / q_max
        zero_point = None
        
        # Quantize
        if group_size > 0:
            w_q = torch.round(w / scale).clamp(-q_max, q_max)
            w_deq = w_q * scale
        else:
            w_q = torch.round(w / scale).clamp(-q_max, q_max)
            w_deq = w_q * scale

    else:
        # Asymmetric: Range is [min, max] mapped to [0, 2^bits - 1]
        if group_size > 0:
            min_val = w.amin(dim=1, keepdim=True)
            max_val = w.amax(dim=1, keepdim=True)
        else:
            min_val = w.amin(dim=1, keepdim=True)
            max_val = w.amax(dim=1, keepdim=True)

        min_val = min_val.clamp(max=0) # Ensure 0 is represented
        max_val = max_val.clamp(min=0)
        
        q_max = 2**bits - 1
        scale = (max_val - min_val) / q_max
        scale = scale.clamp(min=1e-5)
        zero_point = torch.round(-min_val / scale)
        
        # Quantize
        if group_size > 0:
            w_q = torch.round((w / scale) + zero_point).clamp(0, q_max)
            w_deq = (w_q - zero_point) * scale
        else:
            w_q = torch.round((w / scale) + zero_point).clamp(0, q_max)
            w_deq = (w_q - zero_point) * scale

    # Reshape back if grouped
    if group_size > 0:
        w_deq = w_deq.flatten()[:w_orig.numel()].reshape(w_orig.shape)

    return w_deq

def get_mse_loss(w_orig, w_deq):
    return torch.nn.functional.mse_loss(w_orig, w_deq).item()

# ==========================================
# 3. Main Extraction Logic
# ==========================================
def extract_quant_loss():
    print(f"Loading model config: {MODEL_ID}...")
    # We load with device_map="auto" to handle large weights, 
    # but we will process weights on CPU/GPU one by one to save memory.
    try:
        config = AutoConfig.from_pretrained(MODEL_ID)
        # To save time/memory for this demo, we can just load the config and 
        # iterate over layers if we had the weights. 
        # Since we need real weights, we load the model.
        # WARNING: This requires ~90GB VRAM/RAM for full float16 loading.
        # If you are resource constrained, load layer by layer or use `device_map="cpu"`.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="cpu", # Load to CPU RAM to avoid OOM, move specific layers to GPU later
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded. Starting analysis...")
    
    # Storage for results
    # Structure: {config_name: [loss_expert_0, loss_expert_1, ...]}
    top_panel_data = {cfg['name']: [] for cfg in QUANT_CONFIGS}
    
    # Structure: {layer_type: [loss_expert_0, ...]} (Using W4_Channel_Sym as reference)
    bottom_panel_data = {"w1 (Gate)": [], "w2 (Down)": [], "w3 (Up)": []}
    
    # Mixtral has 32 layers, each has 8 experts. 
    # The plot in your image shows ~64 expert IDs. 
    # This likely implies they are flattening experts from the first few layers 
    # OR they are looking at a specific layer across all experts.
    # Let's analyze the first 8 layers * 8 experts = 64 data points.
    
    layers_to_analyze = 8 
    num_experts = config.num_local_experts # Usually 8 for Mixtral
    
    global_expert_id = 0
    
    # Iterate through Transformer Layers
    for layer_idx in tqdm(range(layers_to_analyze), desc="Analyzing Layers"):
        layer = model.model.layers[layer_idx]
        block_sparse_moe = layer.block_sparse_moe
        experts = block_sparse_moe.experts
        
        # Iterate through Experts in this layer
        for expert_idx in range(num_experts):
            expert_layer = experts[expert_idx]
            
            # Mixtral Expert structure:
            # w1: Gate Projection (Hidden Dim -> Intermediate)
            # w2: Down Projection (Intermediate -> Hidden Dim)
            # w3: Up Projection   (Hidden Dim -> Intermediate)
            
            weights = {
                "w1 (Gate)": expert_layer.w1.weight.data.float(), # Convert to float32 for precision
                "w2 (Down)": expert_layer.w2.weight.data.float(),
                "w3 (Up)":   expert_layer.w3.weight.data.float()
            }
            
            # --- Top Panel Calculation (Average loss across all 3 sub-layers) ---
            for cfg in QUANT_CONFIGS:
                total_loss = 0
                for w_name, w_tensor in weights.items():
                    w_deq = quantize_weight(w_tensor, cfg['bits'], cfg['group_size'], cfg['sym'])
                    total_loss += get_mse_loss(w_tensor, w_deq)
                
                # Average loss for this expert under this config
                avg_loss = total_loss / 3 
                top_panel_data[cfg['name']].append(avg_loss)

            # --- Bottom Panel Calculation (Layer breakdown for W4_Channel_Sym) ---
            # We pick one sensitive config to show the breakdown, e.g., W4 Symmetric
            ref_cfg = QUANT_CONFIGS[1] # W4_Channel_Sym
            
            for w_name, w_tensor in weights.items():
                w_deq = quantize_weight(w_tensor, ref_cfg['bits'], ref_cfg['group_size'], ref_cfg['sym'])
                loss = get_mse_loss(w_tensor, w_deq)
                bottom_panel_data[w_name].append(loss)
                
            global_expert_id += 1
            
            # Cleanup to save RAM
            del weights
            torch.cuda.empty_cache()

    return top_panel_data, bottom_panel_data

# ==========================================
# 4. Plotting Function
# ==========================================
def plot_results(top_data, bottom_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.05) # Remove space between plots

    # Plot Top Panel
    markers = ['o', 's', 'v', '^', 'D']
    for i, (name, losses) in enumerate(top_data.items()):
        ax1.plot(losses, label=name, marker=markers[i % len(markers)], markersize=4, alpha=0.8)
    
    ax1.set_ylabel("Quant Loss (MSE)", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12, ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("Weight-Only Quantization Sensitivity by Expert", fontsize=16)

    # Plot Bottom Panel
    colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
    for i, (name, losses) in enumerate(bottom_data.items()):
        ax2.plot(losses, label=name, color=colors[i], marker='o', markersize=4)

    ax2.set_ylabel("Quant Loss (MSE)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Expert ID (Flattened across Layers)", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("mixtral_quant_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    # Note: This will only run if you have the model downloaded/access to HF
    top_data, bottom_data = extract_quant_loss()
    if top_data:
        plot_results(top_data, bottom_data)