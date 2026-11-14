"""RetNet 2.0 decoder-only language model (~300M scale presets + activation checkpointing)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .retention import MultiScaleRetention

Tensor = torch.Tensor
State = list  # one (B, H, dk, dv) tensor or (S, conv_cache) tuple per layer


# Standard size presets (with vocab_size=32000 tied embeddings, 12 d^2 per layer):
#   "300m"      : d_model=1024, n_layers=22, n_heads=16 -> ~277M non-embed, ~310M total params
#   "300m-deep" : d_model=1024, n_layers=24, n_heads=16 -> ~302M non-embed, ~335M total params
#   "125m"      : d_model=768,  n_layers=12, n_heads=12 -> ~85M  non-embed, ~110M total params
#   "30m"       : d_model=512,  n_layers=8,  n_heads=8  -> ~25M  non-embed, ~41M  total params
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "300m": {"d_model": 1024, "n_layers": 22, "n_heads": 16},
    "300m-deep": {"d_model": 1024, "n_layers": 24, "n_heads": 16},
    "125m": {"d_model": 768, "n_layers": 12, "n_heads": 12},
    "30m": {"d_model": 512, "n_layers": 8, "n_heads": 8},
}


@dataclass
class RetNetConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 22
    n_heads: int = 16
    # Paper: values and gate are 2x wider, FFN hidden is 2x d_model. Together
    # that gives 12 d^2 parameters per layer, matching a Transformer layer.
    value_factor: int = 2
    ffn_mult: int = 2
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    # "none" = plain RetNet; "delta" = overwritable delta-rule memory.
    # beta_max=2.0 adds negative eigenvalues in (-1, 1) for state tracking.
    memory: str = "delta"
    beta_max: float = 2.0
    # >0: causal depthwise conv of this width feeding q, k, v (DeltaNet/Mamba-style)
    short_conv: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


def config_from_preset(preset: str, vocab_size: int = 32000, **overrides) -> RetNetConfig:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(MODEL_PRESETS)}")
    kw = {"vocab_size": vocab_size, **MODEL_PRESETS[preset], **overrides}
    return RetNetConfig(**kw)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class RetNetBlock(nn.Module):
    def __init__(self, cfg: RetNetConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.retention = MultiScaleRetention(
            cfg.d_model,
            cfg.n_heads,
            cfg.value_factor,
            cfg.rope_base,
            memory=cfg.memory,
            beta_max=cfg.beta_max,
            short_conv=cfg.short_conv,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_mult * cfg.d_model, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, state, offset, mode, chunk_size, return_state):
        y, new_state = self.retention(
            self.norm1(x), state, offset, mode, chunk_size, return_state
        )
        x = x + self.dropout(y)
        x = x + self.ffn(self.norm2(x))
        return x, new_state


class RetNetLM(nn.Module):
    def __init__(self, cfg: RetNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(RetNetBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        # GPT-2 style: shrink the projections that write into the residual
        # stream so its variance doesn't grow with depth.
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.retention.o_proj.weight, std=resid_std)
            nn.init.normal_(block.ffn.down.weight, std=resid_std)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        """Enable or disable per-layer activation checkpointing during training
        (halves activation memory on 16 GB T4 GPUs for 300M models)."""
        self.grad_checkpointing = bool(enabled)

    def num_params(self, exclude_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        state: State | None = None,
        offset: int | Tensor = 0,
        mode: str = "parallel",
        chunk_size: int = 64,
        return_state: bool = False,
    ) -> tuple[Tensor, Tensor | None, State | None]:
        """Returns (logits, loss, new_state)."""
        x = self.embed(input_ids)
        states = state if state is not None else [None] * len(self.blocks)
        new_states = []
        use_ckpt = (
            self.grad_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and not return_state
            and state is None
        )
        for block, s in zip(self.blocks, states):
            if use_ckpt:
                x = grad_checkpoint(
                    lambda h, b=block: b(h, None, offset, mode, chunk_size, False)[0],
                    x,
                    use_reentrant=False,
                )
                ns = None
            else:
                x, ns = block(x, s, offset, mode, chunk_size, return_state)
            new_states.append(ns)
        logits = self.lm_head(self.norm_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        keep_state = (return_state or mode != "parallel") and not use_ckpt
        return logits, loss, (new_states if keep_state else None)

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        prefill_chunk: int = 256,
    ) -> Tensor:
        """Prefill the prompt chunkwise, then decode one token at a time in
        recurrent mode: constant memory and time per new token, no KV cache.
        """
        logits, _, state = self.forward(prompt, mode="chunkwise", chunk_size=prefill_chunk)
        pos = prompt.shape[1]
        out = [prompt]
        next_logits = logits[:, -1]
        for _ in range(max_new_tokens):
            tok = _sample(next_logits, temperature, top_k)
            out.append(tok)
            if eos_id is not None and bool((tok == eos_id).all()):
                break
            logits, _, state = self.forward(tok, state=state, offset=pos, mode="recurrent")
            pos += 1
            next_logits = logits[:, -1]
        return torch.cat(out, dim=1)


def _sample(logits: Tensor, temperature: float, top_k: int | None) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
