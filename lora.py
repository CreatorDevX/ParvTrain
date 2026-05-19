import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        in_f = linear.in_features
        out_f = linear.out_features

        self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.disable_merge = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        if self.r == 0:
            return base
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    def merge(self):
        if self.r > 0 and not getattr(self, "disable_merge", False):
            self.linear.weight.data.add_((self.lora_B @ self.lora_A) * self.scaling)

    def reset_lora(self, r: Optional[int] = None):
        if getattr(self, "disable_merge", False):
            return
        if r is not None and r != self.r:
            self.r = r
            self.scaling = self.alpha / r if r > 0 else 1.0
            in_f = self.linear.in_features
            out_f = self.linear.out_features
            self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.02)
            self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        else:
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lora_A": self.lora_A, "lora_B": self.lora_B, "scaling": torch.tensor(self.scaling)}

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]):
        self.lora_A.data = state["lora_A"]
        self.lora_B.data = state["lora_B"]
        self.scaling = state["scaling"].item()


class LoRAEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.embedding = embedding
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        vocab = embedding.num_embeddings
        d_model = embedding.embedding_dim

        self.lora_A = nn.Parameter(torch.randn(r, d_model) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(vocab, r))
        self.disable_merge = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.embedding(x)
        if self.r == 0:
            return base
        lora_weight = self.lora_B @ self.lora_A
        lora = F.embedding(x, lora_weight) * self.scaling
        return base + lora

    def merge(self):
        if self.r > 0 and not getattr(self, "disable_merge", False):
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.embedding.weight.data.add_(delta)
            if hasattr(self.embedding, 'weight') and self.embedding.weight is not None:
                pass

    def reset_lora(self, r: Optional[int] = None):
        if getattr(self, "disable_merge", False):
            return
        if r is not None and r != self.r:
            self.r = r
            self.scaling = self.alpha / r if r > 0 else 1.0
            d_model = self.embedding.embedding_dim
            vocab = self.embedding.num_embeddings
            self.lora_A = nn.Parameter(torch.randn(r, d_model) * 0.02)
            self.lora_B = nn.Parameter(torch.zeros(vocab, r))
        else:
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lora_A": self.lora_A, "lora_B": self.lora_B, "scaling": torch.tensor(self.scaling)}

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]):
        self.lora_A.data = state["lora_A"]
        self.lora_B.data = state["lora_B"]
        self.scaling = state["scaling"].item()


def _replace_with_lora(module: nn.Module, r: int = 4, alpha: float = 1.0, skip_names: set = None, weight_map: dict = None):
    skip_names = skip_names or set()
    if weight_map is None:
        weight_map = {}
    for name, child in list(module.named_children()):
        full_name = f"{module._get_name()}.{name}" if hasattr(module, '_get_name') else name
        if any(s in name for s in skip_names):
            continue

        if isinstance(child, nn.Linear):
            weight_id = id(child.weight)
            if weight_id in weight_map:
                existing = weight_map[weight_id]
                lora_layer = LoRALinear(child, r=r, alpha=alpha)
                lora_layer.lora_A = existing.lora_A
                lora_layer.lora_B = existing.lora_B
                lora_layer.scaling = existing.scaling
                lora_layer.disable_merge = True
            else:
                lora_layer = LoRALinear(child, r=r, alpha=alpha)
                weight_map[weight_id] = lora_layer
            setattr(module, name, lora_layer)
        elif isinstance(child, nn.Embedding):
            weight_id = id(child.weight)
            if weight_id in weight_map:
                existing = weight_map[weight_id]
                lora_emb = LoRAEmbedding(child, r=r, alpha=alpha)
                lora_emb.lora_A = existing.lora_A
                lora_emb.lora_B = existing.lora_B
                lora_emb.scaling = existing.scaling
                lora_emb.disable_merge = True
            else:
                lora_emb = LoRAEmbedding(child, r=r, alpha=alpha)
                weight_map[weight_id] = lora_emb
            setattr(module, name, lora_emb)
        else:
            _replace_with_lora(child, r=r, alpha=alpha, skip_names=skip_names, weight_map=weight_map)


def _collect_lora_modules(module: nn.Module) -> list:
    found = []
    for child in module.children():
        if isinstance(child, (LoRALinear, LoRAEmbedding)):
            found.append(child)
        else:
            found.extend(_collect_lora_modules(child))
    return found


def apply_lora(model: nn.Module, r: int = 4, alpha: float = 1.0):
    _replace_with_lora(model, r=r, alpha=alpha)
    return model


def merge_lora(model: nn.Module):
    for mod in _collect_lora_modules(model):
        mod.merge()


def reset_lora(model: nn.Module, r: int = 4, alpha: float = 1.0):
    for mod in _collect_lora_modules(model):
        mod.reset_lora(r)


def pop_lora_weights(model: nn.Module):
    for mod in _collect_lora_modules(model):
        if isinstance(mod, LoRALinear):
            linear = mod.linear
            parent_name, child_name = _find_parent(model, mod)
            if parent_name is not None:
                setattr(parent_name, child_name, linear)
        elif isinstance(mod, LoRAEmbedding):
            emb = mod.embedding
            parent_name, child_name = _find_parent(model, mod)
            if parent_name is not None:
                setattr(parent_name, child_name, emb)


def _find_parent(model: nn.Module, target: nn.Module):
    for name, child in model.named_children():
        if child is target:
            return model, name
        result = _find_parent(child, target)
        if result[0] is not None:
            return result
    return None, None


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for i, mod in enumerate(_collect_lora_modules(model)):
        prefix = f"lora_{i}"
        state[f"{prefix}.lora_A"] = mod.lora_A
        state[f"{prefix}.lora_B"] = mod.lora_B
        state[f"{prefix}.scaling"] = torch.tensor(mod.scaling)
    return state
