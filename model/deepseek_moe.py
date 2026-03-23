# coding=utf-8
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeek MoE model implementation with quantization
# support adapted from the QUIP-style quantized LLaMA implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DeepSeek MoE model with quantization support."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

# Try to import DeepseekConfig from transformers standard models
# If not available, we'll use AutoConfig to load it dynamically or define a minimal version
from model.configuration_deepseek import DeepseekConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

from lib.linear.fused_quantized_linear import FusedQuantizedLinear
from lib.linear.quantized_linear import QuantizedLinear

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return torch.tensor(0.0)

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class DeepseekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DeepseekRMSNorm)


class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekLinearScalingRotaryEmbedding(DeepseekRotaryEmbedding):
    def forward(self, x, position_ids):
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class DeepseekDynamicNTKScalingRotaryEmbedding(DeepseekRotaryEmbedding):
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# =============================================================================
# Quantized MLP (Used for Dense, Shared Experts, and Routed Experts)
# =============================================================================

class DeepseekQuantizedMLP(nn.Module):
    def __init__(self, config: DeepseekConfig, intermediate_size: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        
        quip_params = getattr(config, 'quip_params', None)
        if quip_params is not None:
            self.codesz = quip_params['codesz']
            self.packsz = quip_params.get('packsz', 1)
            self.pack_out = quip_params.get('pack_out', False)
            self.idx_dtype = quip_params['idx_dtype']
            self.codebook_version = quip_params.get('codebook_version', 0)
            self.lora_rank = quip_params['lora_rank']
            self.rescale_WH = quip_params['rescale_WH']
            self.resid_scale_override = quip_params.get('resid_scale_override', -1)
            self.train_mode = quip_params.get('train_mode', False)
        else:
            self.codesz = 8
            self.packsz = 2
            self.pack_out = False
            self.idx_dtype = 'torch.int64'
            self.codebook_version = 1
            self.lora_rank = 0
            self.rescale_WH = False
            self.resid_scale_override = -1
            self.train_mode = False
        
        use_quantized = quip_params is not None and quip_params.get('lora_rank', 0) >= 0
        
        if use_quantized:
            # self.gate_proj = nn.Linear(
            #     self.hidden_size, 
            #     self.intermediate_size, 
            #     bias=False
            # )
            self.gate_proj = QuantizedLinear(
                self.hidden_size, 
                self.intermediate_size, 
                self.codesz, 
                self.packsz, 
                self.pack_out,
                self.idx_dtype, 
                self.codebook_version, 
                rank=self.lora_rank, 
                rescale_WH=self.rescale_WH,
                bias=False, 
                resid_scale_override=self.resid_scale_override, 
                train_mode=self.train_mode,
            )
            # self.up_proj = nn.Linear(
            #     self.hidden_size,
            #     self.intermediate_size,
            #     bias=False
            # )
            self.up_proj = QuantizedLinear(
                self.hidden_size, 
                self.intermediate_size, 
                self.codesz, 
                self.packsz, 
                self.pack_out,
                self.idx_dtype, 
                self.codebook_version, 
                rank=self.lora_rank, 
                rescale_WH=self.rescale_WH,
                bias=False, 
                resid_scale_override=self.resid_scale_override, 
                train_mode=self.train_mode,
            )
            # self.down_proj = nn.Linear(
            #     self.intermediate_size, 
            #     self.hidden_size, 
            #     bias=False
            # )
            self.down_proj = QuantizedLinear(
                self.intermediate_size, 
                self.hidden_size, 
                self.codesz, 
                self.packsz, 
                self.pack_out,
                self.idx_dtype, 
                self.codebook_version, 
                rank=self.lora_rank, 
                rescale_WH=self.rescale_WH,
                bias=False, 
                resid_scale_override=self.resid_scale_override, 
                train_mode=self.train_mode,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        gate_weight = getattr(self.gate_proj, 'weight', None)
        if gate_weight is not None and x.dtype != gate_weight.dtype:
            x = x.to(gate_weight.dtype)
        out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return out.to(input_dtype)


# =============================================================================
# DeepSeek MoE Block
# =============================================================================

class DeepseekMoE(nn.Module):
    """
    DeepSeek Mixture of Experts block.
    Contains routed experts and optional shared experts.
    """
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        
        # Router
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Routed Experts
        self.experts = nn.ModuleList([
            DeepseekQuantizedMLP(config, intermediate_size=config.moe_intermediate_size) 
            for _ in range(self.num_experts)
        ])
        
        # Shared Experts (if any)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            shared_intermediate_size = config.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = DeepseekQuantizedMLP(config, intermediate_size=shared_intermediate_size)
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # 1. Compute Shared Experts
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
        
        # 2. Compute Router Logits
        gate_input = hidden_states
        if gate_input.dtype != self.gate.weight.dtype:
            gate_input = gate_input.to(self.gate.weight.dtype)
        gate_output = self.gate(gate_input)
        # Handle case where gate returns tuple (logits, ...) or just logits
        if isinstance(gate_output, tuple):
            router_logits = gate_output[0]
        else:
            router_logits = gate_output
        # Flatten router_logits for further processing
        router_logits_flat = router_logits.view(-1, router_logits.size(-1))
        routing_weights = F.softmax(router_logits_flat, dim=-1, dtype=torch.float32)
        
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
            
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype) * self.routed_scaling_factor
        
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # 3. Process Routed Experts
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            top_k_idx, idx = torch.where(expert_mask[expert_idx])
            
            if idx.numel() == 0:
                continue
            
            current_hidden_states = hidden_states_flat[idx]
            expert_output = expert(current_hidden_states)
            current_routing_weights = routing_weights_topk[idx, top_k_idx]
            expert_output = expert_output * current_routing_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, idx, expert_output.to(final_hidden_states.dtype))
        
        # 4. Combine Shared and Routed
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
            
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
        return final_hidden_states, router_logits_flat


# =============================================================================
# Attention Modules
# =============================================================================

class DeepseekAttention(nn.Module):
    def __init__(self, config: DeepseekConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        quip_params = getattr(config, 'quip_params', None)
        if quip_params is not None:
            self.codesz = quip_params['codesz']
            self.packsz = quip_params.get('packsz', 1)
            self.pack_out = quip_params.get('pack_out', False)
            self.idx_dtype = quip_params['idx_dtype']
            self.codebook_version = quip_params.get('codebook_version', 0)
            self.lora_rank = quip_params['lora_rank']
            self.rescale_WH = quip_params['rescale_WH']
            self.resid_scale_override = quip_params.get('resid_scale_override', -1)
            self.train_mode = quip_params.get('train_mode', False)
        else:
            self.codesz = 8; self.packsz = 2; self.pack_out = False; self.idx_dtype = 'torch.int64'
            self.codebook_version = 1; self.lora_rank = 0; self.rescale_WH = False
            self.resid_scale_override = -1; self.train_mode = False

        use_quantized = quip_params is not None and quip_params.get('lora_rank', 0) >= 0

        if use_quantized:
            self.q_proj = QuantizedLinear(self.hidden_size, self.num_heads * self.head_dim, self.codesz, self.packsz, self.pack_out, self.idx_dtype, self.codebook_version, rank=self.lora_rank, rescale_WH=self.rescale_WH, resid_scale_override=self.resid_scale_override, train_mode=self.train_mode)
            self.k_proj = QuantizedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.codesz, self.packsz, self.pack_out, self.idx_dtype, self.codebook_version, rank=self.lora_rank, rescale_WH=self.rescale_WH, resid_scale_override=self.resid_scale_override, train_mode=self.train_mode)
            self.v_proj = QuantizedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.codesz, self.packsz, self.pack_out, self.idx_dtype, self.codebook_version, rank=self.lora_rank, rescale_WH=self.rescale_WH, resid_scale_override=self.resid_scale_override, train_mode=self.train_mode)
            self.o_proj = QuantizedLinear(self.num_heads * self.head_dim, self.hidden_size, self.codesz, self.packsz, self.pack_out, self.idx_dtype, self.codebook_version, rank=self.lora_rank, rescale_WH=self.rescale_WH, resid_scale_override=self.resid_scale_override, train_mode=self.train_mode)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()

    def _init_rope(self):
        rope_scaling = getattr(self.config, 'rope_scaling', None)
        if rope_scaling is None:
            self.rotary_emb = DeepseekRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        q_proj_weight = getattr(self.q_proj, 'weight', None)
        if q_proj_weight is not None and hidden_states.dtype != q_proj_weight.dtype:
            hidden_states = hidden_states.to(q_proj_weight.dtype)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if use_cache:
            if past_key_value is None:
                past_key_value = DynamicCache()
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value if use_cache else None

class DeepseekSdpaAttention(DeepseekAttention):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None):
        if output_attentions:
            return super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if use_cache:
            if past_key_value is None:
                past_key_value = DynamicCache()
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] if attention_mask is not None else None
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states, key_states, value_states = query_states.contiguous(), key_states.contiguous(), value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), None, past_key_value if use_cache else None

DEEPSEEK_ATTENTION_CLASSES = {
    "eager": DeepseekAttention,
    "sdpa": DeepseekSdpaAttention,
}


# =============================================================================
# Decoder Layer
# =============================================================================

class DeepseekDecoderLayer(nn.Module):
    def __init__(self, config: DeepseekConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        attn_implementation = getattr(config, '_attn_implementation', 'eager')
        self.self_attn = DEEPSEEK_ATTENTION_CLASSES.get(attn_implementation, DeepseekAttention)(config=config, layer_idx=layer_idx)

        # DeepSeek uses dense MLP for the first few layers
        self.first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        if layer_idx < self.first_k_dense_replace:
            self.mlp = DeepseekQuantizedMLP(config)
        else:
            self.mlp = DeepseekMoE(config)
        
        self.input_layernorm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, output_router_logits=False, use_cache=False, cache_position=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, **kwargs
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        router_logits = None
        if self.layer_idx < self.first_k_dense_replace:
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states, router_logits = self.mlp(hidden_states)
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        if use_cache: outputs += (present_key_value,)
        if output_router_logits: outputs += (router_logits,)
        return outputs


# =============================================================================
# Main Model
# =============================================================================

class DeepseekPreTrainedModel(PreTrainedModel):
    config_class = DeepseekConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()

class DeepseekModel(DeepseekPreTrainedModel):
    def __init__(self, config: DeepseekConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DeepseekDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self): return self.embed_tokens
    def set_input_embeddings(self, value): self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, output_router_logits=None, return_dict=None, cache_position=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", False)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        past_seen_tokens = 0
        if use_cache and past_key_values is not None:
            if isinstance(past_key_values, Cache):
                # Already a Cache object (DynamicCache or StaticCache), use directly
                past_seen_tokens = past_key_values.get_seq_length()
            else:
                # Legacy cache format (tuple/list), convert to DynamicCache
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None: position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids, past_key_value=past_key_values,
                output_attentions=output_attentions, output_router_logits=output_router_logits, use_cache=use_cache, cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if use_cache: next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions: all_self_attns += (layer_outputs[1],)
            if output_router_logits and len(layer_outputs) > (3 if output_attentions else 2):
                if layer_outputs[-1] is not None: all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits] if v is not None)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, router_logits=all_router_logits)

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_seen_tokens):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1: causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        return causal_mask

# =============================================================================
# Causal LM Model
# =============================================================================

class DeepseekForCausalLM(DeepseekPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.001)
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
    def set_decoder(self, decoder): self.model = decoder
    def get_decoder(self): return self.model

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        position_ids=None, 
        past_key_values=None, 
        inputs_embeds=None, 
        labels=None, 
        use_cache=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        output_router_logits=None, 
        return_dict=None, 
        cache_position=None
    ):
        output_router_logits = output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            output_router_logits=output_router_logits, 
            return_dict=return_dict, 
            cache_position=cache_position
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1).to(shift_logits.device))

        aux_loss = None
        if output_router_logits and (outputs.router_logits if return_dict else outputs[-1]):
            aux_loss = load_balancing_loss_func(outputs.router_logits if return_dict else outputs[-1], self.num_experts, self.num_experts_per_tok, attention_mask)
            if labels is not None: loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits: output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache): past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else: past_length = past_key_values[0][0].shape[2]
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]: input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]: input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values: position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids.contiguous()}
        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None: cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else: cache_position = cache_position[-input_length:]

        model_inputs.update({"position_ids": position_ids, "cache_position": cache_position, "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask})
        return model_inputs