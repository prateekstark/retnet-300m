"""Train a RetNet-300M (or 125M/30M) reasoning & coding model on pre-tokenized
Tiny-Series + Reasoning data, with multi-account Shared Google Drive checkpointing.

Example (Colab T4 / L4 / A100, auto batch size, shared Drive checkpoints):
  python scripts/train.py --name retnet-300m --preset 300m \
      --data /content/data/tinymix \
      --out_root /content/drive/MyDrive/retnet-300m/runs \
      --memory delta --beta_max 2 --short_conv 4 \
      --seq_len 1024 --batch_size 0 --seqs_per_step 32 \
      --train_mode chunkwise --chunk_size 128 --resume

Writes to <out_root>/<name>/:
  config.json             model + training arguments
  tokenizer.json          copied once so eval/chat only need the run folder
  log.jsonl               one line per log/eval step
  ckpt_step_XXXXXXX.pt    versioned checkpoints (multi-account safe, rolling keep_ckpts)
  latest.json             pointer to latest completed checkpoint
  samples.txt             generations at each eval
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet import MODEL_PRESETS, RetNetConfig, RetNetLM  # noqa: E402
from retnet.checkpoint import (  # noqa: E402
    compact_optimizer_state,
    load_latest_checkpoint,
    restore_optimizer_state,
    save_checkpoint,
)


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/tinymix"))
    ap.add_argument("--out_root", type=Path, default=Path("runs"))
    # model
    ap.add_argument(
        "--preset",
        choices=list(MODEL_PRESETS),
        default=None,
        help="size preset: 300m (d=1024,L=22,H=16), 300m-deep (d=1024,L=24,H=16), 125m, 30m",
    )
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=22)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--value_factor", type=int, default=2)
    ap.add_argument("--ffn_mult", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--memory", choices=["none", "delta"], default="delta")
    ap.add_argument("--beta_max", type=float, default=2.0)
    ap.add_argument("--short_conv", type=int, default=4)
    ap.add_argument(
        "--train_mode",
        choices=["parallel", "chunkwise"],
        default="chunkwise",
        help="chunkwise: memory grows with T*chunk instead of T^2 (best for 1024+ context)",
    )
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="enable per-layer activation checkpointing (auto-enabled on 16GB GPUs if needed)",
    )
    # optimization
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="sequences per forward pass; 0 = largest that fits this GPU (see --seqs_per_step)",
    )
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument(
        "--seqs_per_step",
        type=int,
        default=32,
        help="with --batch_size 0: sequences per optimizer step; grad_accum is derived",
    )
    ap.add_argument(
        "--sample_prompt",
        default="Question: A baker bakes 7 trays of 24 muffins and sells 150. How many are left?\nAnswer:",
    )
    ap.add_argument("--max_steps", type=int, default=15258)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # bookkeeping & multi-account checkpointing
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=250, help="save checkpoint every N steps")
    ap.add_argument(
        "--save_every_minutes",
        type=float,
        default=15.0,
        help="also save a checkpoint if this many minutes elapsed since last save",
    )
    ap.add_argument("--keep_ckpts", type=int, default=2, help="number of step checkpoints to keep")
    ap.add_argument(
        "--no_compact_ckpt",
        action="store_true",
        help="store optimizer moments in fp32 instead of bfloat16 on disk",
    )
    ap.add_argument(
        "--legacy_ckpt",
        action="store_true",
        help="also write ckpt.pt alongside ckpt_step_XXXXXXX.pt",
    )
    ap.add_argument("--eval_batches", type=int, default=20, help="validation windows = 8 x this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="mixed precision; auto = bf16 on Ampere+, fp16 on Volta/Turing (T4), else fp32",
    )
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (needs Triton, sm_75+)")
    a = ap.parse_args()
    if a.preset is not None:
        for k, v in MODEL_PRESETS[a.preset].items():
            setattr(a, k, v)
    return a


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name != "auto":
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]
    if not device.startswith("cuda"):
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16  # A100 / L4 / RTX 30xx+
    if major >= 7:
        return torch.float16  # V100 / T4: tensor cores, but no fast bf16
    return torch.float32


def require_files(*paths: Path, hint: str) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"error: missing {', '.join(missing)}\n{hint}")


def append_text(path: Path, text: str) -> None:
    """Append to a log file across accounts; if Drive FUSE denies append on a
    file owned by another account, copy-and-replace or write to a session log."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
            f.flush()
        return
    except OSError:
        pass
    try:
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(existing + text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        print(f"warning: could not append to {path}: {e}", flush=True)


def check_and_write_heartbeat(run_dir: Path, step: int) -> None:
    """Detect if another Colab account/runtime is actively training the same run."""
    hb_path = run_dir / "heartbeat.json"
    session_id = f"{socket.gethostname()}:{os.getpid()}"
    now = time.time()
    if hb_path.is_file():
        try:
            prev = json.loads(hb_path.read_text())
            if (
                prev.get("session") != session_id
                and now - float(prev.get("timestamp", 0)) < 90
            ):
                print(
                    f"WARNING: another session ({prev.get('session')}) updated {run_dir.name} "
                    f"{now - float(prev.get('timestamp', 0)):.0f}s ago at step {prev.get('step')}! "
                    "Make sure two Colab notebooks are not training the same run simultaneously.",
                    flush=True,
                )
        except Exception:
            pass
    try:
        hb_path.write_text(
            json.dumps({"session": session_id, "step": step, "timestamp": now}, indent=2)
        )
    except OSError:
        pass


class TokenData:
    def __init__(self, path: Path, seq_len: int, device: str, vocab_size: int):
        try:
            self.data = np.memmap(path, dtype=np.uint16, mode="r")
            max_id = int(self.data.max()) if len(self.data) else -1
        except (OSError, ValueError) as e:
            sys.exit(f"error: could not read token file {path}: {e!r}")
        if len(self.data) <= seq_len + 1:
            sys.exit(f"error: {path} has {len(self.data)} tokens, need more than seq_len+1")
        if max_id >= vocab_size:
            sys.exit(
                f"error: {path} contains token id {max_id} but vocab_size is "
                f"{vocab_size}; re-run scripts/prepare_mix.py"
            )
        self.seq_len = seq_len
        self.device = device

    def batch_at(self, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.seq_len
        chunk = np.stack([self.data[s : s + T + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(chunk)
        if self.device.startswith("cuda"):
            t = t.pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]

    def random_batch(self, rng: np.random.Generator, bs: int):
        return self.batch_at(rng.integers(0, len(self.data) - self.seq_len - 1, size=bs))


def pick_batch_size(model: RetNetLM, a: argparse.Namespace, amp, vocab_size: int, headroom=0.84) -> int:
    """Largest power-of-two micro-batch dividing --seqs_per_step whose training
    step fits in `headroom` of GPU memory.

    If a ~300M model on a 16 GB T4 can only fit batch_size <= 2 without
    activation checkpointing, automatically enables `model.set_grad_checkpointing(True)`
    and re-probes so it can run with a faster micro-batch.
    """
    total = torch.cuda.get_device_properties(a.device).total_memory
    adam = 8 * sum(p.numel() for p in model.parameters())
    candidates = [
        b for b in (64, 32, 16, 8, 4, 2, 1) if b <= a.seqs_per_step and a.seqs_per_step % b == 0
    ]

    def probe() -> int | None:
        for bs in candidates:
            x = torch.randint(0, vocab_size, (bs, a.seq_len), device=a.device)
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(a.device)
                with amp():
                    _, loss, _ = model(x, x, mode=a.train_mode, chunk_size=a.chunk_size)
                loss.backward()
                peak = torch.cuda.max_memory_allocated(a.device) + adam
                fits = peak < headroom * total
            except torch.OutOfMemoryError:
                fits, peak = False, None
            finally:
                model.zero_grad(set_to_none=True)
                loss = x = None
                torch.cuda.empty_cache()
            used = f"{peak / 2**30:.1f}" if peak else "OOM"
            print(
                f"  batch {bs:3d} (grad_ckpt={model.grad_checkpointing}): "
                f"{used} / {total / 2**30:.1f} GB {'-> use' if fits else ''}",
                flush=True,
            )
            if fits:
                return bs
        return None

    best = probe()
    if (best is None or best < 4) and not model.grad_checkpointing:
        print("  enabling activation checkpointing (--grad_ckpt) to fit larger micro-batch ...", flush=True)
        model.set_grad_checkpointing(True)
        best_ckpt = probe()
        if best_ckpt is not None and (best is None or best_ckpt > best):
            return best_ckpt
        if best is not None:
            model.set_grad_checkpointing(False)
            return best
    if best is not None:
        return best
    sys.exit("error: even batch size 1 does not fit; use a smaller --seq_len or model")


def lr_at(step: int, a: argparse.Namespace) -> float:
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    progress = min(1.0, (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps))
    min_lr = a.lr * a.min_lr_ratio
    return min_lr + 0.5 * (a.lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, a: argparse.Namespace):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.dim() < 2 or "embed" in name else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": a.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, val: TokenData, starts: np.ndarray, bs: int, amp) -> float:
    """Loss on a fixed set of validation windows, so runs are comparable."""
    model.eval()
    losses = []
    for i in range(0, len(starts), bs):
        x, y = val.batch_at(starts[i : i + bs])
        with amp():
            _, loss, _ = model(x, y, mode="chunkwise", chunk_size=128)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample_text(model, tokenizer, device, prompt: str, n: int = 160) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    out = model.generate(ids, max_new_tokens=n, temperature=0.8, top_k=50, eos_id=0)
    model.train()
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    a = get_args()
    torch.manual_seed(a.seed)
    out = a.out_root / a.name
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"error: cannot create run directory {out}: {e}")

    from tokenizers import Tokenizer

    require_files(
        *(a.data / f for f in ("meta.json", "tokenizer.json", "train.bin", "val.bin")),
        hint="run scripts/prepare_mix.py first, or pass --data",
    )
    try:
        meta = json.loads((a.data / "meta.json").read_text())
        # Only copy tokenizer.json if not already present (avoids cross-account overwrite on Shared Drive)
        run_tok = out / "tokenizer.json"
        if not run_tok.is_file():
            shutil.copyfile(a.data / "tokenizer.json", run_tok)
        vocab_size = int(meta["vocab_size"])
        tokenizer = Tokenizer.from_file(str(a.data / "tokenizer.json"))
    except Exception as e:
        sys.exit(
            f"error: could not load meta.json/tokenizer.json from {a.data}: {e!r}\n"
            "re-run scripts/prepare_mix.py"
        )

    cfg = RetNetConfig(
        vocab_size=vocab_size,
        d_model=a.d_model,
        n_layers=a.n_layers,
        n_heads=a.n_heads,
        value_factor=a.value_factor,
        ffn_mult=a.ffn_mult,
        dropout=a.dropout,
        memory=a.memory,
        beta_max=a.beta_max,
        short_conv=a.short_conv,
    )
    raw_model = RetNetLM(cfg).to(a.device)
    if a.grad_ckpt:
        raw_model.set_grad_checkpointing(True)
    opt = make_optimizer(raw_model, a)

    dtype = resolve_dtype(a.dtype, a.device)
    device_type = "cuda" if a.device.startswith("cuda") else "cpu"

    def amp():
        if dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device_type, dtype=dtype)

    scaler = torch.amp.GradScaler(device_type, enabled=dtype == torch.float16)
    print(f"precision: {str(dtype).replace('torch.', '')}")

    if a.batch_size <= 0:
        if a.seqs_per_step <= 0 or not a.device.startswith("cuda"):
            sys.exit("error: --batch_size 0 (auto) needs --seqs_per_step and a CUDA device")
        print(f"finding the largest batch that fits ({a.seqs_per_step} sequences per step):")
        a.batch_size = pick_batch_size(raw_model, a, amp, vocab_size)
        a.grad_accum = a.seqs_per_step // a.batch_size
        print(
            f"batch {a.batch_size} x grad_accum {a.grad_accum} "
            f"(grad_ckpt={raw_model.grad_checkpointing})",
            flush=True,
        )

    step = 0
    ck, ck_path = (load_latest_checkpoint(out, a.device) if a.resume else (None, None))
    if a.resume and ck is None:
        print(f"--resume: no checkpoint found in {out}, starting from scratch", flush=True)
    if ck is not None and ck_path is not None:
        try:
            raw_model.load_state_dict(ck["model"])
            restore_optimizer_state(opt, ck["optimizer"], a.device)
            # Cross-GPU safe GradScaler load: only load if scaler is enabled on THIS GPU
            # AND the checkpoint actually recorded a non-empty fp16 scaler state.
            ck_scaler = ck.get("scaler")
            if scaler.is_enabled() and isinstance(ck_scaler, dict) and "scale" in ck_scaler:
                scaler.load_state_dict(ck_scaler)
            step = int(ck["step"])
        except Exception as e:
            sys.exit(
                f"error: could not resume from {ck_path}: {e!r}\n"
                "(different model flags than the original run?)"
            )
        print(f"resumed from {ck_path.name} at step {step}", flush=True)
    else:
        cfg_file = out / "config.json"
        if not cfg_file.is_file():
            cfg_dump = {
                "model": cfg.to_dict(),
                "train": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            }
            try:
                cfg_file.write_text(json.dumps(cfg_dump, indent=2))
            except OSError as e:
                sys.exit(f"error: cannot write to run directory {out}: {e}")

    check_and_write_heartbeat(out, step)

    train = TokenData(a.data / "train.bin", a.seq_len, a.device, vocab_size)
    val = TokenData(a.data / "val.bin", a.seq_len, a.device, vocab_size)
    eval_starts = np.random.default_rng(1234).integers(
        0, len(val.data) - a.seq_len - 1, size=a.eval_batches * 8
    )
    rng = np.random.default_rng(a.seed + step)
    tokens_per_step = a.batch_size * a.grad_accum * a.seq_len

    print(
        f"params: {raw_model.num_params(False) / 1e6:.2f}M total, "
        f"{raw_model.num_params(True) / 1e6:.2f}M non-embedding"
    )
    print(
        f"tokens/step: {tokens_per_step}, total: {tokens_per_step * a.max_steps / 1e6:.0f}M",
        flush=True,
    )

    def do_save(cur_step: int) -> None:
        opt_sd = (
            opt.state_dict()
            if a.no_compact_ckpt
            else compact_optimizer_state(opt.state_dict())
        )
        saved_p = save_checkpoint(
            {
                "model": raw_model.state_dict(),
                "optimizer": opt_sd,
                "scaler": scaler.state_dict() if scaler.is_enabled() else {},
                "step": cur_step,
                "config": cfg.to_dict(),
            },
            out,
            keep_ckpts=a.keep_ckpts,
            also_write_legacy_ckpt=a.legacy_ckpt,
        )
        if saved_p is not None:
            check_and_write_heartbeat(out, cur_step)

    # Graceful preemption / stop signal handler for Colab account switching
    stop_requested = False

    def _on_signal(signum, _frame):
        nonlocal stop_requested
        sig_name = signal.Signals(signum).name
        print(f"\n[{sig_name}] caught; will save checkpoint after current step and exit cleanly ...", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    model = torch.compile(raw_model) if a.compile else raw_model
    model.train()
    t_start = t_last = t_last_save = time.time()
    running, n_running = 0.0, 0

    while step < a.max_steps:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr

        for _ in range(a.grad_accum):
            x, y = train.random_batch(rng, a.batch_size)
            with amp():
                _, loss, _ = model(x, y, mode=a.train_mode, chunk_size=a.chunk_size)
            scaler.scale(loss / a.grad_accum).backward()
            running += loss.item() / a.grad_accum
        n_running += 1
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), a.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        now = time.time()
        time_to_save = (
            (a.save_every > 0 and step % a.save_every == 0)
            or (a.save_every_minutes > 0 and (now - t_last_save) >= a.save_every_minutes * 60)
            or (step % a.eval_every == 0)
            or (step == a.max_steps)
            or stop_requested
        )

        if step % a.log_every == 0 or step == a.max_steps or stop_requested:
            rec = {
                "step": step,
                "tokens": step * tokens_per_step,
                "train_loss": running / max(1, n_running),
                "lr": lr,
                "grad_norm": gnorm,
                "tok_per_s": (
                    a.log_every * tokens_per_step / max(1e-6, now - t_last)
                    if step % a.log_every == 0
                    else None
                ),
                "elapsed_s": now - t_start,
            }
            running, n_running, t_last = 0.0, 0, now
            if step % a.eval_every == 0 or step == a.max_steps:
                rec["val_loss"] = evaluate(model, val, eval_starts, a.batch_size, amp)
                text = sample_text(
                    raw_model, tokenizer, a.device, a.sample_prompt.replace("\\n", "\n")
                )
                append_text(
                    out / "samples.txt",
                    f"=== step {step} val_loss {rec['val_loss']:.4f}\n{text}\n\n",
                )
            append_text(out / "log.jsonl", json.dumps(rec) + "\n")
            check_and_write_heartbeat(out, step)
            msg = f"step {step:6d} | loss {rec['train_loss']:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f}"
            if rec["tok_per_s"]:
                msg += f" | {rec['tok_per_s'] / 1e3:.1f}k tok/s"
            if "val_loss" in rec:
                msg += f" | val {rec['val_loss']:.4f}"
            print(msg, flush=True)

        if time_to_save:
            do_save(step)
            t_last_save = time.time()

        if stop_requested:
            try:
                os.sync()
            except AttributeError:
                pass
            print(f"saved checkpoint at step {step} and exiting cleanly.", flush=True)
            return

    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
