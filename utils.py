import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from config import ModelConfig


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {"total": total, "trainable": trainable}
    tag_params: Dict[str, int] = {}

    for name, param in model.named_parameters():
        n = param.numel()
        if "routed_experts" in name:
            tag_params["routed_expert"] = tag_params.get("routed_expert", 0) + n
        elif "shared_expert" in name:
            tag_params["shared_expert"] = tag_params.get("shared_expert", 0) + n
        elif "global_tokens" in name:
            tag_params["attention_global"] = tag_params.get("attention_global", 0) + n
        elif "router" in name:
            tag_params["router"] = tag_params.get("router", 0) + n
        elif "gate_proj" in name:
            tag_params["dense_swiglu"] = tag_params.get("dense_swiglu", 0) + n
        elif ("up_proj" in name or "down_proj" in name) and ".ffn." in name and "shared_expert" not in name and "routed_experts" not in name:
            tag_params["dense_swiglu"] = tag_params.get("dense_swiglu", 0) + n
        elif any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            tag_params["attention"] = tag_params.get("attention", 0) + n
        elif "norm" in name:
            tag_params["rmsnorm"] = tag_params.get("rmsnorm", 0) + n
        elif "token_embedding" in name:
            tag_params["embedding"] = tag_params.get("embedding", 0) + n
        elif "lm_head" in name:
            tag_params["lm_head"] = tag_params.get("lm_head", 0) + n
        else:
            tag_params["other"] = tag_params.get("other", 0) + n

    breakdown["breakdown"] = tag_params
    return breakdown


def estimate_memory(
    config: ModelConfig,
    batch_size: int = 1,
    seq_len: int = 32768,
    dtype: torch.dtype = torch.float16,
    include_grads: bool = True,
    include_optim: bool = True,
) -> Dict[str, float]:
    bytes_per_param = 2 if dtype == torch.float16 else 4

    n_params_embed = config.vocab_size * config.d_model
    n_params_attn_per_layer = (
        config.d_model * config.n_heads * config.d_head
        + config.d_model * config.n_kv_heads * config.d_head * 2
        + config.n_heads * config.d_head * config.d_model
    )
    n_params_attn = n_params_attn_per_layer * config.n_layers

    n_params_dense_ffn = (
        3 * config.d_model * config.d_ff * config.n_dense_layers
    )

    n_params_shared_expert = (
        2 * config.d_model * config.d_ff * config.n_moe_layers
    )
    n_params_routed_expert = (
        2 * config.d_model * config.d_ff * config.moe.n_routed_experts * config.n_moe_layers
    )
    n_params_router = config.d_model * config.moe.n_routed_experts * config.n_moe_layers
    n_params_norms = (config.n_layers * 2 + 1) * config.d_model
    n_params_global_tokens = config.n_global_tokens * config.d_model

    total_params = (
        n_params_embed
        + n_params_attn
        + n_params_dense_ffn
        + n_params_shared_expert
        + n_params_routed_expert
        + n_params_router
        + n_params_norms
        + n_params_global_tokens
    )

    param_mb = total_params * bytes_per_param / (1024 * 1024)

    activation_memory_per_token = 0
    activation_memory_per_token += config.n_layers * config.n_heads * config.d_head * 4
    activation_memory_per_token += config.n_layers * config.d_model * 8
    activation_mb = activation_memory_per_token * batch_size * seq_len * bytes_per_param / (1024 * 1024)

    n_kv_cache_slots = batch_size * config.n_kv_heads * config.d_head * min(seq_len, config.max_seq_len)
    kv_cache_bytes = 1 if config.use_kv_8bit else bytes_per_param
    kv_cache_mb = n_kv_cache_slots * kv_cache_bytes * config.n_layers * 2 / (1024 * 1024)

    total_mb = param_mb
    grad_mb = 0
    optim_mb = 0

    if include_grads:
        grad_mb = param_mb
        total_mb += grad_mb

    if include_optim:
        if dtype == torch.float16:
            optim_mb = param_mb * 2
        else:
            optim_mb = param_mb * 3
        total_mb += optim_mb

    total_mb += activation_mb + kv_cache_mb

    routing_info = {
        "n_layers": config.n_layers,
        "n_moe_layers": config.n_moe_layers,
        "n_dense_layers": config.n_dense_layers,
        "n_routed_experts_per_layer": config.moe.n_routed_experts,
        "n_shared_experts_per_layer": config.moe.n_shared_experts,
        "top_k": config.moe.top_k,
        "capacity_factor": config.moe.capacity_factor,
        "n_active_experts_per_token": config.moe.top_k + config.moe.n_shared_experts,
    }

    return {
        "total_params": total_params,
        "param_mb": round(param_mb, 2),
        "activation_mb": round(activation_mb, 2),
        "kv_cache_mb": round(kv_cache_mb, 2),
        "gradient_mb": round(grad_mb, 2),
        "optimizer_mb": round(optim_mb, 2),
        "estimated_total_mb": round(total_mb, 2),
        "estimated_total_gb": round(total_mb / 1024, 2),
        "active_params_per_token": round(
            (
                config.n_layers * config.d_model * (
                    config.n_heads * config.d_head  # Q
                    + 2 * config.n_kv_heads * config.d_head  # K, V
                    + config.n_heads * config.d_head  # O
                )  # attention all layers
                + config.n_dense_layers * 3 * config.d_model * config.d_ff  # dense SwiGLU
                + config.n_moe_layers * 2 * config.d_model * config.d_ff  # shared experts
                + config.n_moe_layers * config.moe.top_k * 2 * config.d_model * config.d_ff  # routed top-k
                + config.n_moe_layers * config.d_model * config.moe.n_routed_experts  # routers
                + (config.n_layers * 2 + 1) * config.d_model  # norms
            ) / 1e6, 2
        ),
        "routing_info": routing_info,
    }


def print_model_summary(model: nn.Module, config: ModelConfig):
    counts = count_parameters(model)
    mem = estimate_memory(config)

    print("=" * 60)
    print("MoE Pretrain Model Summary")
    print("=" * 60)
    print(f"Total Parameters:     {counts['total']:>12,}")
    print(f"Trainable Parameters: {counts['trainable']:>12,}")
    print(f"Model Size (FP16):    {mem['param_mb']:>10.2f} MB")
    print(f"KV Cache (8-bit):     {mem['kv_cache_mb']:>10.2f} MB")
    print(f"Activations:          {mem['activation_mb']:>10.2f} MB")
    print(f"Estimated Total:      {mem['estimated_total_mb']:>10.2f} MB")
    print(f"                      {mem['estimated_total_gb']:>10.2f} GB")
    print("-" * 60)
    print("Breakdown:")
    for tag, n in counts["breakdown"].items():
        if n > 0:
            pct = n / counts["total"] * 100
            print(f"  {tag:<20s} {n:>10,} ({pct:>5.1f}%)")
    print("-" * 60)
    print("Routing Info:")
    ri = mem["routing_info"]
    print(f"  MoE Layers:          {ri['n_moe_layers']}")
    print(f"  Dense Layers:        {ri['n_dense_layers']}")
    print(f"  Routed Experts/Layer:{ri['n_routed_experts_per_layer']}")
    print(f"  Top-K:               {ri['top_k']}")
    print(f"  + Shared Expert:     {ri['n_shared_experts_per_layer']}")
    print(f"  Active/Layer/Token:  {ri['n_active_experts_per_token']}")
    print(f"  Active Params/Token: {mem['active_params_per_token']}M")
    print("=" * 60)
