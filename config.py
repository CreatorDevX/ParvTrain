from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from transformers import PretrainedConfig


@dataclass
class RoPEScalingConfig:
    type: str = "yarn"
    factor: float = 8.0
    original_max_len: int = 4096
    attention_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1


@dataclass
class MoEConfig:
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    top_k: int = 1
    capacity_factor: float = 1.25
    aux_loss_alpha: float = 0.01
    z_loss_coeff: float = 1e-4
    router_jitter_noise: float = 0.1
    router_init_std: float = 0.01
    drop_overflow: bool = True


@dataclass
class ModelConfig:
    # ── ~120M total params / ~17.2M active params (excl. embedding) ──
    # Total breakdown:
    #   Embedding (tied):  12.6M
    #   Attention (10L):    3.9M
    #   Dense FFN (2L):     1.5M
    #   Shared expert (8L): 5.9M
    #   Routed 16×8L:      95.2M
    #   Router+norms+glob:  0.1M
    # Active per token:
    #   Attention: 3.9M + Dense: 1.5M + Shared: 5.9M + top-1 routed: 5.9M = ~17.2M
    vocab_size: int = 32768
    d_model: int = 384
    n_layers: int = 10
    n_moe_layers: int = 8
    n_dense_layers: int = 2
    n_heads: int = 6
    n_kv_heads: int = 2
    d_head: int = 64
    d_ff: int = 650
    activation: str = "swiglu"
    rope_base: int = 10000
    rope_scaling: Optional[RoPEScalingConfig] = None
    max_seq_len: int = 32768
    sliding_window: int = 1024
    n_global_tokens: int = 32
    use_flash_attn: bool = True
    tie_word_embeddings: bool = True
    moe: MoEConfig = field(default_factory=MoEConfig)
    use_kv_8bit: bool = True

    @property
    def n_dense_layer_indices(self) -> tuple:
        return (0, self.n_layers - 1)

    @property
    def n_moe_layer_indices(self) -> range:
        return range(1, self.n_layers - 1)


class ParvHFConfig(PretrainedConfig):
    model_type = "parv"

    def __init__(
        self,
        vocab_size=32000,
        d_model=384,
        n_layers=10,
        n_moe_layers=8,
        n_dense_layers=2,
        n_heads=6,
        n_kv_heads=2,
        d_head=64,
        d_ff=768,
        activation="swiglu",
        rope_base=10000,
        rope_scaling=None,
        max_seq_len=32768,
        sliding_window=4096,
        n_global_tokens=32,
        use_flash_attn=True,
        tie_word_embeddings=True,
        moe=None,
        use_kv_8bit=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_moe_layers = n_moe_layers
        self.n_dense_layers = n_dense_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.activation = activation
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.max_seq_len = max_seq_len
        self.sliding_window = sliding_window
        self.n_global_tokens = n_global_tokens
        self.use_flash_attn = use_flash_attn
        self.tie_word_embeddings = tie_word_embeddings
        self.moe = moe or {}
        self.use_kv_8bit = use_kv_8bit

    def to_model_config(self) -> ModelConfig:
        mc = ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_moe_layers=self.n_moe_layers,
            n_dense_layers=self.n_dense_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
            d_ff=self.d_ff,
            activation=self.activation,
            rope_base=self.rope_base,
            max_seq_len=self.max_seq_len,
            sliding_window=self.sliding_window,
            n_global_tokens=self.n_global_tokens,
            use_flash_attn=self.use_flash_attn,
            tie_word_embeddings=self.tie_word_embeddings,
            use_kv_8bit=self.use_kv_8bit,
        )
        if self.rope_scaling is not None:
            mc.rope_scaling = RoPEScalingConfig(**self.rope_scaling)
        if self.moe:
            mc.moe = MoEConfig(**self.moe)
        return mc
