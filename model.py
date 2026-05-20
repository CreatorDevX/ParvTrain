import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_yarn_freqs(
    dim: int,
    max_len: int,
    base: float = 10000.0,
    scale: float = 1.0,
    original_max_len: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    dims = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (dims / dim))

    if scale > 1.0:
        n_dims = dim // 2
        dim_vals = torch.arange(n_dims, dtype=torch.float32, device=device)
        ratio = max_len / original_max_len

        low = dim_vals / n_dims < beta_slow / (beta_slow + beta_fast)
        high = dim_vals / n_dims > beta_fast / (beta_slow + beta_fast)
        mid = ~(low | high)

        ramp = torch.zeros(n_dims, dtype=torch.float32, device=device)
        ramp[low] = 0.0
        ramp[high] = 1.0
        if mid.any():
            ramp[mid] = (dim_vals[mid] / n_dims - beta_slow / (beta_slow + beta_fast)) / (
                beta_fast / (beta_slow + beta_fast) - beta_slow / (beta_slow + beta_fast)
            )

        scaled_base = base * scale ** (dim / (dim - 2))
        inv_freq_ntk = 1.0 / (scaled_base ** (dims / dim))

        interp_inv_freq = inv_freq / ratio

        inv_freq = (1.0 - ramp) * interp_inv_freq + ramp * inv_freq_ntk

    return inv_freq


class YaRNRoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_head = config.d_head
        self.base = config.rope_base
        self.max_seq_len = config.max_seq_len

        scaling = config.rope_scaling
        if scaling is not None:
            self.scale = scaling.factor
            self.original_max_len = scaling.original_max_len
            self.beta_fast = scaling.beta_fast
            self.beta_slow = scaling.beta_slow
            self.attention_factor = scaling.attention_factor
        else:
            self.scale = 1.0
            self.original_max_len = config.max_seq_len
            self.beta_fast = 32
            self.beta_slow = 1
            self.attention_factor = 1.0

        inv_freq = precompute_yarn_freqs(
            dim=self.d_head,
            max_len=self.max_seq_len,
            base=self.base,
            scale=self.scale,
            original_max_len=self.original_max_len,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float()
        pos = position_ids[:, None, :].float()
        freqs = pos * inv_freq
        cos = freqs.cos().transpose(1, 2)
        sin = freqs.sin().transpose(1, 2)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_rot = x * cos + rotate_half(x) * sin
    return x_rot


class KVCache:
    def __init__(
        self,
        max_batch_size: int = 1,
        max_seq_len: int = 32768,
        n_kv_heads: int = 2,
        d_head: int = 64,
        quantize_8bit: bool = True,
        n_global_tokens: int = 0,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.quantize_8bit = quantize_8bit
        self.n_global_tokens = n_global_tokens
        self.clear()

    def clear(self):
        self.cache_k = None
        self.cache_v = None
        self.k_scales = None
        self.v_scales = None
        self.seq_len = 0

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = abs_max / 127.0
        quant = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return quant, scale

    def dequantize(self, quant: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (quant.float() * scale).to(dtype)

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_heads, seq_len, d_head = k.shape

        if self.cache_k is None:
            if self.quantize_8bit:
                self.cache_k = torch.zeros(
                    batch_size, n_heads, self.max_seq_len, d_head,
                    dtype=torch.int8, device=k.device,
                )
                self.cache_v = torch.zeros_like(self.cache_k)
                self.k_scales = torch.zeros(batch_size, n_heads, self.max_seq_len, 1, device=k.device)
                self.v_scales = torch.zeros_like(self.k_scales)
            else:
                self.cache_k = torch.zeros(
                    batch_size, n_heads, self.max_seq_len, d_head,
                    dtype=k.dtype, device=k.device,
                )
                self.cache_v = torch.zeros_like(self.cache_k)

        new_len = self.seq_len + seq_len
        if new_len > self.max_seq_len:
            new_len = self.max_seq_len
            overflow = self.seq_len + seq_len - self.max_seq_len
            self._evict(overflow)

        if self.quantize_8bit:
            k_q, k_s = self.quantize(k)
            v_q, v_s = self.quantize(v)
            self.cache_k[:, :, self.seq_len : new_len] = k_q
            self.cache_v[:, :, self.seq_len : new_len] = v_q
            self.k_scales[:, :, self.seq_len : new_len] = k_s
            self.v_scales[:, :, self.seq_len : new_len] = v_s
            full_k = self.dequantize(self.cache_k[:, :, :new_len], self.k_scales[:, :, :new_len], k.dtype)
            full_v = self.dequantize(self.cache_v[:, :, :new_len], self.v_scales[:, :, :new_len], v.dtype)
        else:
            self.cache_k[:, :, self.seq_len : new_len] = k
            self.cache_v[:, :, self.seq_len : new_len] = v
            full_k = self.cache_k[:, :, :new_len]
            full_v = self.cache_v[:, :, :new_len]

        self.seq_len = new_len
        return full_k, full_v

    def _evict(self, n_tokens: int):
        g = self.n_global_tokens
        if g > 0:
            self.cache_k[:, :, g : -n_tokens] = self.cache_k[:, :, g + n_tokens :].clone()
            self.cache_v[:, :, g : -n_tokens] = self.cache_v[:, :, g + n_tokens :].clone()
            self.cache_k[:, :, -n_tokens:] = 0
            self.cache_v[:, :, -n_tokens:] = 0
            if self.quantize_8bit:
                self.k_scales[:, :, g : -n_tokens] = self.k_scales[:, :, g + n_tokens :].clone()
                self.v_scales[:, :, g : -n_tokens] = self.v_scales[:, :, g + n_tokens :].clone()
                self.k_scales[:, :, -n_tokens:] = 0
                self.v_scales[:, :, -n_tokens:] = 0
        else:
            self.cache_k = torch.roll(self.cache_k, shifts=-n_tokens, dims=2)
            self.cache_v = torch.roll(self.cache_v, shifts=-n_tokens, dims=2)
            self.cache_k[:, :, -n_tokens:] = 0
            self.cache_v[:, :, -n_tokens:] = 0
            if self.quantize_8bit:
                self.k_scales = torch.roll(self.k_scales, shifts=-n_tokens, dims=2)
                self.v_scales = torch.roll(self.v_scales, shifts=-n_tokens, dims=2)
                self.k_scales[:, :, -n_tokens:] = 0
                self.v_scales[:, :, -n_tokens:] = 0
        self.seq_len -= n_tokens


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_heads // config.n_kv_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.sliding_window = config.sliding_window
        self.n_global_tokens = config.n_global_tokens

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        self.rope = YaRNRoPE(config)

        self.kv_cache: Optional[KVCache] = None

    def init_kv_cache(self, max_batch_size: int = 1, max_seq_len: Optional[int] = None):
        self.kv_cache = KVCache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len or self.config.max_seq_len,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
            quantize_8bit=self.config.use_kv_8bit,
            n_global_tokens=self.n_global_tokens,
        )

    def clear_kv_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.clear()

    def _build_sliding_window_mask(
        self,
        q_len: int,
        kv_len: int,
        past_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.full((q_len, kv_len), float("-inf"), dtype=dtype, device=device)
        
        q_idx = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(0, kv_len, device=device).unsqueeze(0)
        
        # 1. Global tokens are always allowed to be attended to
        is_global = kv_idx < self.n_global_tokens
        
        # 2. Local tokens are allowed if causal and within sliding window
        causal = kv_idx <= q_idx
        in_window = (q_idx - kv_idx) < self.sliding_window
        
        # 3. For global queries (q_idx < G), they can only attend to global tokens
        # For local queries (q_idx >= G), they can attend to global tokens + local causal/window tokens
        q_is_global = q_idx < self.n_global_tokens
        
        # Global queries: only attend to global tokens
        global_query_mask = q_is_global & is_global
        
        # Local queries: attend to global tokens OR (local causal/window tokens)
        local_query_mask = (~q_is_global) & (is_global | (causal & in_window))
        
        allowed = global_query_mask | local_query_mask
        mask[allowed] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(x, position_ids)
        q = apply_rotary_emb(q, cos, sin)
        k_rope = apply_rotary_emb(k, cos, sin)

        # Scale query by attention_factor for YaRN RoPE scaling
        if hasattr(self.rope, "attention_factor") and self.rope.attention_factor != 1.0:
            q = q * math.sqrt(self.rope.attention_factor)

        if use_kv_cache and self.kv_cache is not None:
            k_rope, v = self.kv_cache.update(k_rope, v)

        k_rope = k_rope.repeat_interleave(self.n_kv_groups, dim=1)
        v = v.repeat_interleave(self.n_kv_groups, dim=1)

        past_len = k_rope.size(2) - seq_len
        attn_mask = self._build_sliding_window_mask(
            q_len=seq_len,
            kv_len=k_rope.size(2),
            past_len=past_len,
            device=device,
            dtype=x.dtype,
        )

        out = F.scaled_dot_product_attention(
            q, k_rope, v, attn_mask=attn_mask, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        moe = config.moe

        self.shared_expert = ExpertFFN(config)
        self.routed_experts = nn.ModuleList(
            [ExpertFFN(config) for _ in range(moe.n_routed_experts)]
        )
        self.router = nn.Linear(config.d_model, moe.n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=moe.router_init_std)
        self.register_buffer("expert_counts", torch.zeros(moe.n_routed_experts, dtype=torch.long))

    def forward(
        self,
        x: torch.Tensor,
        update_aux_loss: bool = False,
        global_step: int = 0,
        warmup_steps: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        moe = self.config.moe
        orig_shape = x.shape
        x_flat = x.view(-1, self.config.d_model)
        n_tokens = x_flat.size(0)
        n_experts = moe.n_routed_experts

        router_logits = self.router(x_flat)

        if self.training and global_step < warmup_steps and update_aux_loss:
            noise = torch.rand_like(router_logits)
            gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
            router_logits = router_logits + gumbel_noise * moe.router_jitter_noise

        routing_weights = F.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)

        capacity = math.ceil(
            n_tokens * moe.top_k / n_experts * moe.capacity_factor
        )
        if capacity < 1:
            capacity = 1

        expert_mask = F.one_hot(top_k_indices, num_classes=n_experts).sum(dim=1)
        tokens_per_expert = expert_mask.sum(dim=0)
        if self.training:
            self.expert_counts.add_(tokens_per_expert)

        final_output = torch.zeros_like(x_flat)

        for expert_idx in range(n_experts):
            mask = top_k_indices == expert_idx
            selected = mask.any(dim=-1)
            selected_indices = torch.where(selected)[0]

            if len(selected_indices) > capacity and moe.drop_overflow:
                selected_indices = selected_indices[:capacity]

            if len(selected_indices) > 0:
                expert_input = x_flat[selected_indices]
                expert_out = self.routed_experts[expert_idx](expert_input)

                if moe.top_k == 1:
                    w = top_k_weights[selected_indices, 0].unsqueeze(-1)
                    final_output[selected_indices] += expert_out * w
                else:
                    for k in range(moe.top_k):
                        expert_k_mask = top_k_indices[selected_indices, k] == expert_idx
                        if expert_k_mask.any():
                            w = top_k_weights[selected_indices, k][expert_k_mask].unsqueeze(-1)
                            final_output[selected_indices[expert_k_mask]] += expert_out[expert_k_mask] * w

        shared_out = self.shared_expert(x_flat)
        final_output = final_output + shared_out

        aux_loss = x.new_zeros(1)
        if self.training and update_aux_loss:
            router_probs = F.softmax(router_logits.float(), dim=-1)
            # Vectorized frequency calculation avoiding nested Python loops
            frac_tokens = F.one_hot(top_k_indices, num_classes=n_experts).float().mean(dim=(0, 1))

            router_prob_mean = router_probs.mean(dim=0)
            load_balance_loss = (
                (frac_tokens * router_prob_mean).sum() * n_experts
            )
            aux_loss = load_balance_loss * moe.aux_loss_alpha

            z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()
            aux_loss = aux_loss + z_loss * moe.z_loss_coeff

        return final_output.view(orig_shape), aux_loss


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, is_moe: bool):
        super().__init__()
        self.is_moe = is_moe
        self.self_attn = Attention(config, layer_idx)
        self.input_norm = RMSNorm(config.d_model)
        self.post_attn_norm = RMSNorm(config.d_model)

        if is_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        update_aux_loss: bool = False,
        global_step: int = 0,
        warmup_steps: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(x, position_ids, attention_mask, use_kv_cache)
        x = residual + x

        residual = x
        x = self.post_attn_norm(x)
        if self.is_moe:
            x, aux_loss = self.ffn(x, update_aux_loss, global_step, warmup_steps)
        else:
            x = self.ffn(x)
            aux_loss = x.new_zeros(1)
        x = residual + x

        return x, aux_loss


class ParvModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        if config.n_global_tokens > 0:
            self.global_tokens = nn.Parameter(
                torch.randn(1, config.n_global_tokens, config.d_model) * 0.02
            )

        self.layers = nn.ModuleList()
        dense_indices = config.n_dense_layer_indices

        for idx in range(config.n_layers):
            is_moe = idx in config.n_moe_layer_indices
            self.layers.append(DecoderLayer(config, idx, is_moe))

        self.norm = RMSNorm(config.d_model)

        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        n_layers = self.config.n_layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "router" not in name and "lm_head" not in name:
                    std = 0.02
                    # Scale down output projections to stabilize deep residual propagation
                    if "o_proj" in name or "down_proj" in name:
                        std = 0.02 / math.sqrt(2 * n_layers)
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_kv_caches(self, max_batch_size: int = 1, max_seq_len: Optional[int] = None):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(max_batch_size, max_seq_len)

    def clear_kv_caches(self):
        for layer in self.layers:
            layer.self_attn.clear_kv_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        update_aux_loss: bool = False,
        global_step: int = 0,
        warmup_steps: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden = self.token_embedding(input_ids)

        is_prefill_or_training = (seq_len > 1) or (not use_kv_cache)

        if self.config.n_global_tokens > 0 and is_prefill_or_training:
            g_tokens = self.global_tokens.expand(batch_size, -1, -1).to(hidden.dtype)
            hidden = torch.cat([g_tokens, hidden], dim=1)
            
            g_pos = torch.arange(self.config.n_global_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
            if position_ids is None:
                position_ids = torch.arange(
                    self.config.n_global_tokens, 
                    self.config.n_global_tokens + seq_len, 
                    device=device
                ).unsqueeze(0).expand(batch_size, -1)
            position_ids = torch.cat([g_pos, position_ids], dim=1)
        else:
            if position_ids is None:
                if use_kv_cache:
                    past_len = self.layers[0].self_attn.kv_cache.seq_len if self.layers[0].self_attn.kv_cache is not None else 0
                    position_ids = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                else:
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        total_aux_loss = hidden.new_zeros(1)
        use_checkpoint = getattr(self, "gradient_checkpointing", False) and self.training
        for layer in self.layers:
            if use_checkpoint:
                def create_custom_forward(target_layer):
                    def custom_forward(h, pos_ids, attn_mask):
                        return target_layer(
                            h,
                            pos_ids,
                            attn_mask,
                            use_kv_cache,
                            update_aux_loss,
                            global_step,
                            warmup_steps,
                        )
                    return custom_forward

                hidden, aux_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden, aux_loss = layer(
                    hidden,
                    position_ids,
                    attention_mask,
                    use_kv_cache,
                    update_aux_loss,
                    global_step,
                    warmup_steps,
                )
            total_aux_loss = total_aux_loss + aux_loss

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        return {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "hidden_states": hidden,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        self.clear_kv_caches()
        batch_size = input_ids.size(0)
        self.init_kv_caches(max_batch_size=batch_size)

        generated = input_ids.clone()
        # Prefill: run the full prompt through the model to populate KV cache
        out = self.forward(input_ids, position_ids=None, use_kv_cache=True, update_aux_loss=False)

        for _ in range(max_new_tokens):
            logits = out["logits"][:, -1, :]

            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

            out = self.forward(
                next_token,
                position_ids=None,
                use_kv_cache=True,
                update_aux_loss=False,
            )

        self.clear_kv_caches()
        return generated
