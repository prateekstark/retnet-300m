"""Generate a high-resolution architecture diagram for RetNet-300M (RetNet 2.0)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = "#1e293b",
    ec: str = "#38bdf8",
    tc: str = "#f8fafc",
    sc: str = "#cbd5e1",
    fontsize: float = 10.0,
    subsize: float = 8.2,
    lw: float = 1.6,
    radius: float = 0.12,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    cx = x + w / 2.0
    if subtitle:
        ax.text(
            cx,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            color=tc,
            fontsize=fontsize,
            fontweight="bold",
            zorder=zorder + 1,
        )
        ax.text(
            cx,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            color=sc,
            fontsize=subsize,
            zorder=zorder + 1,
        )
    else:
        ax.text(
            cx,
            y + h / 2.0,
            title,
            ha="center",
            va="center",
            color=tc,
            fontsize=fontsize,
            fontweight="bold",
            zorder=zorder + 1,
        )


def arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "#94a3b8",
    lw: float = 1.6,
    rad: float = 0.0,
    style: str = "-|>",
    zorder: int = 2,
):
    conn = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        connectionstyle=conn,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arr)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retnet_architecture.png"

    fig, ax = plt.subplots(figsize=(18, 11), dpi=200)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Title Banner
    ax.text(
        9.0,
        10.55,
        "RetNet-300M (RetNet 2.0) Architecture: Delta-Rule Memory + Short Conv + Calculator-in-the-Loop",
        ha="center",
        va="center",
        color="#f8fafc",
        fontsize=15.5,
        fontweight="bold",
    )
    ax.text(
        9.0,
        10.18,
        "d_model = 1024  |  n_layers = 22  |  n_heads = 16 (dk = 64, dv = 128)  |  vocab = 32,000 (Tied)  |  ~310M Total Parameters",
        ha="center",
        va="center",
        color="#38bdf8",
        fontsize=10.5,
    )

    # =========================================================================
    # PANEL 1 (Left): Full Language Model Stack + Calculator Loop (x: 0.4 .. 5.4)
    # =========================================================================
    panel1 = FancyBboxPatch(
        (0.35, 0.35),
        5.1,
        9.45,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#111827",
        edgecolor="#334155",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(panel1)
    ax.text(
        2.9,
        9.5,
        "1. RetNet-300M Decoder Stack",
        ha="center",
        va="center",
        color="#38bdf8",
        fontsize=12,
        fontweight="bold",
    )

    # Bottom-up in Panel 1
    box(
        ax,
        0.75,
        0.65,
        4.3,
        0.75,
        "Digit-Split BPE Tokenizer (vocab = 32,000)",
        "Digits '0'-'9' & '<<', '>>', '=' isolated  |  uint16 stream",
        fc="#1e293b",
        ec="#64748b",
    )
    arrow(ax, 2.9, 1.40, 2.9, 1.75)

    box(
        ax,
        0.75,
        1.75,
        4.3,
        0.70,
        "Token Embedding  W_E",
        "Shape: (B, T, 1024)  |  Tied with LM Head",
        fc="#1e293b",
        ec="#38bdf8",
    )
    arrow(ax, 2.9, 2.45, 2.9, 2.85)

    # Outer 22x RetNetBlock container
    block_frame = FancyBboxPatch(
        (0.65, 2.85),
        4.5,
        4.35,
        boxstyle="round,pad=0.04,rounding_size=0.15",
        facecolor="#0f172a",
        edgecolor="#f59e0b",
        linewidth=1.8,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(block_frame)
    ax.text(
        2.9,
        6.98,
        "× 22  RetNetBlock  (12 d² params/layer)",
        ha="center",
        va="center",
        color="#fbbf24",
        fontsize=10.5,
        fontweight="bold",
        zorder=4,
    )

    box(
        ax,
        1.0,
        3.05,
        3.8,
        0.52,
        "Pre-LayerNorm (norm1)",
        "eps = 1e-5",
        fc="#1e293b",
        ec="#475569",
        fontsize=9.2,
    )
    arrow(ax, 2.9, 3.57, 2.9, 3.82)

    box(
        ax,
        1.0,
        3.82,
        3.8,
        0.92,
        "Multi-Scale Retention (RetNet 2.0)",
        "ShortConv(K=4) + RoPE + Delta Memory (β<2)",
        fc="#1e3a5f",
        ec="#38bdf8",
        tc="#e0f2fe",
        fontsize=9.6,
    )
    arrow(ax, 2.9, 4.74, 2.9, 5.02)

    # Residual 1
    ax.text(
        4.92,
        4.0,
        "+ Residual",
        color="#94a3b8",
        fontsize=8.0,
        rotation=90,
        va="center",
        ha="center",
    )

    box(
        ax,
        1.0,
        5.02,
        3.8,
        0.52,
        "Pre-LayerNorm (norm2) + Residual",
        "x = x + Retention(norm1(x))",
        fc="#1e293b",
        ec="#475569",
        fontsize=9.2,
    )
    arrow(ax, 2.9, 5.54, 2.9, 5.80)

    box(
        ax,
        1.0,
        5.80,
        3.8,
        0.88,
        "Feed-Forward Network (FFN)",
        "Linear(1024→2048) → GELU → Linear(2048→1024)",
        fc="#1e293b",
        ec="#a855f7",
        fontsize=9.5,
    )

    arrow(ax, 2.9, 7.20, 2.9, 7.52)

    box(
        ax,
        0.75,
        7.52,
        4.3,
        0.68,
        "Final LayerNorm + Tied LM Head",
        "1024 → 32,000 Vocabulary Logits",
        fc="#1e293b",
        ec="#38bdf8",
    )
    arrow(ax, 2.9, 8.20, 2.9, 8.50)

    box(
        ax,
        0.75,
        8.50,
        4.3,
        0.75,
        "AST Calculator-in-the-Loop",
        "On '<<expr=' → safe_eval(expr) → inject 'result>>' in O(1)",
        fc="#064e3b",
        ec="#34d399",
        tc="#ecfdf5",
        sc="#a7f3d0",
    )

    # =========================================================================
    # PANEL 2 (Center): MultiScaleRetention + Delta Rule Internals (x: 5.75 .. 12.15)
    # =========================================================================
    panel2 = FancyBboxPatch(
        (5.70, 0.35),
        6.5,
        9.45,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#111827",
        edgecolor="#334155",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(panel2)
    ax.text(
        8.95,
        9.5,
        "2. Inside Multi-Scale Retention (Delta Memory + Short Conv)",
        ha="center",
        va="center",
        color="#38bdf8",
        fontsize=12,
        fontweight="bold",
    )

    # Input X
    box(
        ax,
        7.15,
        0.65,
        3.6,
        0.65,
        "Normalized Input  x = LayerNorm(X)",
        "Shape: (B, T, 1024)",
        fc="#1e293b",
        ec="#64748b",
    )

    # Arrows from x to ShortConv, Beta proj, and Swish Gate
    arrow(ax, 8.2, 1.30, 7.55, 1.75)
    arrow(ax, 9.6, 1.30, 10.85, 3.05)
    arrow(ax, 10.75, 0.98, 11.75, 6.45, color="#f472b6", rad=0.22)

    # Short Conv
    box(
        ax,
        6.0,
        1.75,
        3.5,
        0.82,
        "Short Causal Depthwise Conv (K=4) + SiLU",
        "Mixes [conv_cache(K-1), x] so keys bind local context",
        fc="#1e3a5f",
        ec="#38bdf8",
        fontsize=9.2,
    )

    # Projections Q, K, V from ShortConv
    arrow(ax, 6.7, 2.57, 6.55, 3.05)
    arrow(ax, 7.75, 2.57, 8.05, 3.05)
    arrow(ax, 8.8, 2.57, 9.45, 3.05)

    box(
        ax,
        5.9,
        3.05,
        1.35,
        0.85,
        "Q Proj",
        "1024→1024\nRoPE + L2Norm",
        fc="#1e293b",
        ec="#38bdf8",
        fontsize=9.0,
        subsize=7.5,
    )
    box(
        ax,
        7.38,
        3.05,
        1.35,
        0.85,
        "K Proj",
        "1024→1024\nRoPE + L2Norm",
        fc="#1e293b",
        ec="#38bdf8",
        fontsize=9.0,
        subsize=7.5,
    )
    box(
        ax,
        8.86,
        3.05,
        1.25,
        0.85,
        "V Proj",
        "1024→2048\n(2× value_factor)",
        fc="#1e293b",
        ec="#38bdf8",
        fontsize=9.0,
        subsize=7.5,
    )
    box(
        ax,
        10.22,
        3.05,
        1.72,
        0.85,
        "Write Strength β_t",
        "β_max · σ(W_b x)\nβ_t ∈ (0, 2.0) per head",
        fc="#3b1d5e",
        ec="#c084fc",
        tc="#f3e8ff",
        fontsize=8.8,
        subsize=7.4,
    )

    # Arrows into FP32 Retention + Delta Core
    arrow(ax, 6.55, 3.90, 7.2, 4.35)
    arrow(ax, 8.05, 3.90, 8.2, 4.35)
    arrow(ax, 9.48, 3.90, 9.1, 4.35)
    arrow(ax, 11.05, 3.90, 10.1, 4.35, color="#c084fc")

    # Retention + Delta-Rule Core Box
    core_box = FancyBboxPatch(
        (6.0, 4.35),
        5.1,
        2.50,
        boxstyle="round,pad=0.03,rounding_size=0.14",
        facecolor="#0f2942",
        edgecolor="#22d3ee",
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(core_box)
    ax.text(
        8.55,
        6.55,
        "Retention + Delta-Rule Memory Core (FP32 Precision)",
        ha="center",
        va="center",
        color="#67e8f9",
        fontsize=10.2,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        8.55,
        6.10,
        "Fixed Multi-Scale Head Decays:   γ_h = 1 − 2^(−5 − h),   h = 0..15",
        ha="center",
        va="center",
        color="#e2e8f0",
        fontsize=8.6,
        zorder=4,
    )
    ax.text(
        8.55,
        5.58,
        "Delta Erase-Before-Write State Update (overwrites old key bindings):",
        ha="center",
        va="center",
        color="#fde68a",
        fontsize=8.6,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        8.55,
        5.15,
        "u_t = β_t (v_t − γ_h k_t S_{t−1}) ,     S_t = γ_h S_{t−1} + k_t^T u_t",
        ha="center",
        va="center",
        color="#ffffff",
        fontsize=8.6,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        8.55,
        4.65,
        "Readout: o_t = q_t S_t   |   Block Form: (I + diag(β) A) u = diag(β)(v − ξ (k S_0))",
        ha="center",
        va="center",
        color="#cbd5e1",
        fontsize=7.5,
        zorder=4,
    )

    arrow(ax, 8.55, 6.85, 8.55, 7.25)

    # GroupNorm per head
    box(
        ax,
        6.20,
        7.25,
        4.6,
        0.65,
        "Per-Head Normalization (GroupNorm / LayerNorm, dv=128)",
        "Rescales each head's different decay horizon γ_h",
        fc="#1e293b",
        ec="#38bdf8",
        fontsize=8.5,
    )

    # Swish Gate box on the right
    box(
        ax,
        10.90,
        6.45,
        1.12,
        1.15,
        "Swish Gate\nG Proj",
        "SiLU(W_g x)\n1024→2048",
        fc="#3b1d38",
        ec="#f472b6",
        tc="#fce7f3",
        fontsize=8.6,
        subsize=7.4,
    )
    arrow(ax, 8.55, 7.90, 8.55, 8.38)
    arrow(ax, 11.45, 7.60, 9.95, 8.70, color="#f472b6", rad=0.15)

    # Elementwise Gating + Output Proj
    box(
        ax,
        6.55,
        8.38,
        4.2,
        0.82,
        "Swish Gating & Output Projection  W_O",
        "y = W_O ( SiLU(W_g x)  ⊙  GroupNorm(o) )   [2048 → 1024]",
        fc="#1e293b",
        ec="#38bdf8",
        fontsize=9.5,
    )

    # =========================================================================
    # PANEL 3 (Right): Three Equivalent Modes + Multi-Account Colab (x: 12.5 .. 17.65)
    # =========================================================================
    panel3 = FancyBboxPatch(
        (12.45, 0.35),
        5.2,
        9.45,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#111827",
        edgecolor="#334155",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(panel3)
    ax.text(
        15.05,
        9.5,
        "3. Triple Mode & Shared Drive Pipeline",
        ha="center",
        va="center",
        color="#38bdf8",
        fontsize=12,
        fontweight="bold",
    )

    # 3 Equivalent Modes
    box(
        ax,
        12.75,
        7.85,
        4.6,
        1.25,
        "A. Chunkwise Training Mode  (O(T · B), chunk=128)",
        "• Parallel triangular delta-solve inside each 128-token chunk\n"
        "• Recurrent state S_[i] passed across chunks  →  Linear memory in T",
        fc="#1e293b",
        ec="#38bdf8",
        fontsize=9.3,
        subsize=8.0,
    )
    box(
        ax,
        12.75,
        6.35,
        4.6,
        1.25,
        "B. Recurrent Inference Mode  (O(1) per token)",
        "• Constant state:  S ∈ R^(B×16×64×128) + conv_cache ∈ R^(B×3×1024)\n"
        "• Zero KV-cache growth! Batched GSM8K & code decoding (64× parallel)",
        fc="#1e293b",
        ec="#34d399",
        fontsize=9.3,
        subsize=8.0,
    )
    box(
        ax,
        12.75,
        5.05,
        4.6,
        1.05,
        "C. Parallel Mode  (O(T²))",
        "• Single-block special case (T ≤ chunk_size)\n"
        "• Tested to match Recurrent & Chunkwise to ~1e-9 in float64",
        fc="#1e293b",
        ec="#a855f7",
        fontsize=9.3,
        subsize=8.0,
    )

    # Divider line in Panel 3
    ax.plot([12.75, 17.35], [4.75, 4.75], color="#334155", lw=1.2, ls="--")
    ax.text(
        15.05,
        4.45,
        "Tiny-Series Data & Multi-Account Shared Drive",
        ha="center",
        va="center",
        color="#fbbf24",
        fontsize=10.8,
        fontweight="bold",
    )

    box(
        ax,
        12.75,
        2.65,
        4.6,
        1.52,
        "100% Textbooks, Code & Reasoning Mix (1B Tokens)",
        "• tiny-codes (24%) + tiny-code-textbooks (12%) + Magicoder (4%)\n"
        "• tiny-math-textbooks (20%) + MetaMathQA (12%) + GSM8K (2%)\n"
        "• tiny-textbooks (14%) + tiny-orca-textbooks (10%) + tiny-lessons (2%)",
        fc="#1e293b",
        ec="#fbbf24",
        fontsize=9.1,
        subsize=7.7,
    )

    box(
        ax,
        12.75,
        0.65,
        4.6,
        1.75,
        "Multi-Account Shared Google Drive Checkpointing",
        "• Staging: Local SSD /tmp → ckpt_step_XXXXXXX.pt + os.fsync\n"
        "• Cross-Account ACL Safe: Each account writes new step files\n"
        "• Quota Safe: 0-byte truncate before unlink + bf16 AdamW moments\n"
        "• Auto-Fallback: Skips truncated latest file on Colab disconnect",
        fc="#1e293b",
        ec="#f43f5e",
        fontsize=9.1,
        subsize=7.6,
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved architecture diagram to {out_path}")


if __name__ == "__main__":
    main()
