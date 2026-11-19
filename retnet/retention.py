"""Multi-scale retention (Sun et al., 2023, "Retentive Network") with
Delta-Rule Memory and Short Convolution (RetNet 2.0).

One layer, three equivalent ways to compute it:

  parallel   : O = (Q K^T * D) V                 -- training, O(T^2) like attention
  recurrent  : S_n = g S_{n-1} + k_n^T u_n,
               o_n = q_n S_n                     -- inference, O(1) per token
  chunkwise  : parallel inside a chunk, recurrent across chunks
                                                 -- long sequences, O(T * chunk)

D[n, m] = g^(n - m) for n >= m else 0, with one fixed decay g per head.

Shapes used throughout:
  B batch, T time, H heads, dk key dim per head, dv value dim per head.
  q, k : (B, H, T, dk)     v : (B, H, T, dv)     state : (B, H, dk, dv)

Why Delta Memory + Short Conv for Coding & Reasoning:
  * Plain RetNet uses an additive update S_n = g S_{n-1} + k_n^T v_n with
    g in (0, 1). Old key-value bindings only fade and can never be overwritten,
    and positive decays cannot track parity or loop counters (Grazzi et al., 2024).
  * memory="delta" uses the delta-rule update:
      S_n = g S_{n-1} + beta_n k_n^T (v_n - g k_n S_{n-1})
    which erases the old value associated with key k_n before writing v_n.
  * With beta_max=2.0, the transition (I - beta k k^T) admits negative
    eigenvalues in (-1, 1), enabling state tracking, parity, and variable
    reassignment in code/reasoning traces.
  * short_conv=4 applies a causal depthwise convolution before q, k, v so keys
    can bind local token pairs (e.g. variable identifier + '=' + value).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------
# Rotary position embedding
# --------------------------------------------------------------------------
def rotary_cos_sin(
    offset: int | Tensor, length: int, dim: int, base: float, device, dtype
) -> tuple[Tensor, Tensor]:
    """cos/sin tables for absolute positions offset..offset+length-1.

    offset is an int (tables of shape (length, dim)) or a (B,) tensor with one
    start position per sequence (tables of shape (B, 1, length, dim)), which
    batched decoding of prompts of different lengths needs.
    """
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=calc_dtype) / dim)
    )
    steps = torch.arange(length, device=device, dtype=calc_dtype)
    if isinstance(offset, Tensor):
        pos = offset.to(device=device, dtype=calc_dtype)[:, None, None] + steps  # (B, 1, length)
    else:
        pos = steps + offset
    freqs = pos[..., None] * inv_freq  # (..., length, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (..., length, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate pairs (x_i, x_{i+dim/2}). x: (B, H, T, dim); cos/sin: (T, dim) or (B, 1, T, dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# --------------------------------------------------------------------------
# Decay
# --------------------------------------------------------------------------
def retnet_log_gamma(n_heads: int) -> Tensor:
    """log g_h with g_h = 1 - 2^(-5 - h * 4 / n_heads) or paper's multi-scale schedule.

    Note: In the RetNet paper (Sun et al., 2023), gamma = 1 - 2^(-5 - arange(0, n_heads)).
    """
    h = torch.arange(n_heads, dtype=torch.float64)
    return torch.log1p(-torch.pow(2.0, -5.0 - h))


# --------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------
class MultiScaleRetention(nn.Module):
    MODES = ("parallel", "recurrent", "chunkwise")

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        value_factor: int = 2,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        memory: str = "delta",
        beta_max: float = 2.0,
        short_conv: int = 4,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        if memory not in ("none", "delta"):
            raise ValueError(f"memory must be 'none' or 'delta', got {memory!r}")
        self.memory = memory
        self.beta_max = beta_max
        self.n_heads = n_heads
        self.head_k = d_model // n_heads
        assert self.head_k % 2 == 0, "head dim must be even for RoPE"
        self.head_v = self.head_k * value_factor
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.scale = self.head_k**-0.5

        d_v = self.head_v * n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)
        self.g_proj = nn.Linear(d_model, d_v, bias=False)  # swish gate
        self.o_proj = nn.Linear(d_v, d_model, bias=False)
        if memory == "delta":
            self.b_proj = nn.Linear(d_model, n_heads, bias=True)  # write strength
        self.short_conv = short_conv
        if short_conv > 0:
            self.conv = nn.Conv1d(d_model, d_model, short_conv, groups=d_model, bias=False)

        # Buffer, so it follows .to(device). Not saved in checkpoints: it is
        # fully determined by n_heads.
        self.register_buffer("log_gamma", retnet_log_gamma(n_heads), persistent=False)

    # ---- helpers ---------------------------------------------------------
    def _log_gamma(self, dtype) -> Tensor:
        return self.log_gamma.to(dtype).view(-1, 1, 1)  # (H, 1, 1)

    def _split_heads(self, x: Tensor, head_dim: int) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, head_dim).transpose(1, 2)

    def _mix(self, x: Tensor, cache: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Causal depthwise conv over [previous K-1 inputs, x]; returns the
        mixed sequence and the new cache (the last K-1 inputs)."""
        if self.short_conv == 0:
            return x, None
        K = self.short_conv
        if cache is None:
            cache = x.new_zeros(x.shape[0], K - 1, x.shape[2])
        inp = torch.cat([cache.to(x.dtype), x], dim=1)  # (B, K-1+T, d)
        mixed = self.conv(inp.transpose(1, 2)).transpose(1, 2)  # (B, T, d)
        return F.silu(mixed), inp[:, inp.shape[1] - (K - 1) :]

    def _project(self, x: Tensor, offset: int | Tensor, mixed: Tensor):
        q = self._split_heads(self.q_proj(mixed), self.head_k)
        k = self._split_heads(self.k_proj(mixed), self.head_k)
        v = self._split_heads(self.v_proj(mixed), self.head_v)
        cos, sin = rotary_cos_sin(
            offset, x.shape[1], self.head_k, self.rope_base, x.device, x.dtype
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if self.memory == "none":
            return q, k * self.scale, v, None
        # Delta rule needs unit-norm keys: (I - beta k k^T) is then a proper
        # (partial) projection, and the memory stays bounded.
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2) * self.beta_max  # (B,H,T)
        return q, k, v, beta

    def _output(self, x: Tensor, o: Tensor) -> Tensor:
        # Per-head normalization (GroupNorm with one group per head).
        o = F.layer_norm(o, (self.head_v,), eps=self.norm_eps)
        B, H, T, dv = o.shape
        o = o.transpose(1, 2).reshape(B, T, H * dv)
        return self.o_proj(F.silu(self.g_proj(x)) * o)

    # ---- the three computations -----------------------------------------
    def _chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor | None,
        need_state: bool,
        beta: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Retention over one block of T tokens, given the state before it.

        With state=None this is exactly the parallel form.
        """
        T = q.shape[-2]
        lg = self._log_gamma(q.dtype)  # (H, 1, 1)
        idx = torch.arange(T, device=q.device)
        diff = (idx[:, None] - idx[None, :]).to(q.dtype)  # n - m
        # Clamp before exp so g^(negative) never overflows; masked to 0 anyway.
        decay = torch.exp(diff.clamp(min=0) * lg).masked_fill(diff < 0, 0.0)  # (H,T,T)
        pos = idx.to(q.dtype)

        if beta is not None:
            v = self._delta_values(k, v, state, beta, decay, diff, lg, pos)

        o = (q @ k.transpose(-1, -2) * decay) @ v  # inside the block

        if state is not None:
            # Query at block position i sees the old state decayed i+1 times.
            q_decay = torch.exp((pos + 1) * lg.squeeze(-1))  # (H, T)
            o = o + (q @ state) * q_decay[..., None]

        new_state = None
        if need_state:
            # Key at block position j has decayed (T-1-j) times by the block end.
            k_decay = torch.exp((T - 1 - pos) * lg.squeeze(-1))  # (H, T)
            new_state = (k * k_decay[..., None]).transpose(-1, -2) @ v
            if state is not None:
                new_state = new_state + state * torch.exp(T * lg)
        return o, new_state

    def _delta_values(self, k, v, state, beta, decay, diff, lg, pos):
        """Turn a delta-rule block into a plain retention block via a batched
        unit-lower-triangular solve."""
        A = (k @ k.transpose(-1, -2)) * decay.masked_fill(diff == 0, 0.0)  # strictly lower
        rhs = v
        if state is not None:
            xi = torch.exp((pos + 1) * lg.squeeze(-1))  # (H, T): g^(t+1)
            rhs = v - xi[..., None] * (k @ state)
        rhs = beta[..., None] * rhs
        M = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device) + beta[..., None] * A
        return torch.linalg.solve_triangular(M, rhs, upper=False, unitriangular=True)

    def _recurrent(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor | None,
        beta: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        B, H, T, _ = q.shape
        if state is None:
            state = q.new_zeros(B, H, self.head_k, self.head_v)
        gamma = torch.exp(self._log_gamma(q.dtype))  # (H, 1, 1)
        outs = []
        for t in range(T):
            vt = v[:, :, t, None, :]  # (B, H, 1, dv)
            if beta is not None:  # write only the error vs. what is stored under k
                stored = k[:, :, t, None, :] @ state
                vt = beta[:, :, t, None, None] * (vt - gamma * stored)
            state = gamma * state + k[:, :, t, :, None] * vt  # outer product
            outs.append(q[:, :, t, None, :] @ state)  # (B, H, 1, dv)
        return torch.cat(outs, dim=2), state

    # ---- public ------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        offset: int | Tensor = 0,
        mode: str = "parallel",
        chunk_size: int = 64,
        return_state: bool = False,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """x: (B, T, d_model). ``offset`` is the absolute position of x[:, 0].

        ``mask`` (B, T), 1 for real tokens and 0 for left padding: padded
        positions write nothing to the state and feed zeros to the short conv,
        so a left-padded sequence gives the same state and outputs (at real
        positions) as the unpadded one. Retention and RoPE only depend on
        relative positions, so the shift from padding does not matter.
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        S, cache = state if isinstance(state, tuple) else (state, None)
        if mask is not None:
            m = mask.to(x.dtype)
            x = x * m[..., None]
        mixed, new_cache = self._mix(x, cache)
        q, k, v, beta = self._project(x, offset, mixed)
        if mask is not None:
            mh = m[:, None, :, None]  # (B, 1, T, 1)
            k, v = k * mh, v * mh
            if beta is not None:
                beta = beta * m[:, None, :]
        state = S

        # Run the retention core in fp32 so decayed sums and the triangular
        # delta solve never overflow or lose precision under fp16/bf16.
        low = (torch.float16, torch.bfloat16)
        if q.dtype in low or k.dtype in low or v.dtype in low:
            q, k, v = q.float(), k.float(), v.float()
            beta = beta.float() if beta is not None else None
            state = state.float() if state is not None else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            o, new_state = self._retention(q, k, v, beta, state, mode, chunk_size, return_state)

        if new_state is not None and new_cache is not None:
            new_state = (new_state, new_cache)
        return self._output(x, o), new_state

    def _retention(self, q, k, v, beta, state, mode, chunk_size, return_state):
        if mode == "parallel":
            o, new_state = self._chunk(q, k, v, state, need_state=return_state, beta=beta)
        elif mode == "recurrent":
            o, new_state = self._recurrent(q, k, v, state, beta=beta)
        else:
            outs, new_state = [], state
            for s in range(0, q.shape[2], chunk_size):
                sl = slice(s, s + chunk_size)
                o_c, new_state = self._chunk(
                    q[:, :, sl],
                    k[:, :, sl],
                    v[:, :, sl],
                    new_state,
                    need_state=True,
                    beta=beta[:, :, sl] if beta is not None else None,
                )
                outs.append(o_c)
            o = torch.cat(outs, dim=2)
        return o, new_state
