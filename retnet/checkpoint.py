"""Multi-account Shared Google Drive safe checkpointing and run loading.

Why special checkpoint handling is required for shared Google Drive folders
across multiple Google Colab accounts:
  1. Cross-account file ownership (ACL):
     When Account A creates `ckpt.pt` in a shared Drive folder, Account A is
     the owner. When Account B mounts the same folder as an Editor, Google
     Drive FUSE often forbids Account B from unlinking or `os.replace`-ing a
     file owned by Account A (`PermissionError`).
     -> Solution: Every checkpoint is saved as a new step-numbered file
        `ckpt_step_0001000.pt` (owned by whichever account wrote that step),
        staged via local SSD (`/tmp`) first and flushed with `os.fsync`.
  2. Interrupted Drive syncs on Colab disconnect:
     If Account A disconnects while a 2-3 GB checkpoint is still syncing to
     Drive, the newest file may be truncated.
     -> Solution: `load_latest_checkpoint` scans all `ckpt_step_*.pt` (and
        `ckpt.pt`) in descending step order and automatically falls back to
        the previous valid checkpoint if the newest one is unreadable.
  3. Google Drive 15 GB quota & Trash bloat:
     Deleting multi-GB checkpoints moves them to Drive Trash, which still
     counts against the owner's 15 GB quota; and non-owners cannot unlink
     files owned by another account.
     -> Solution: Before unlinking old checkpoints beyond `keep_ckpts`, we
        first truncate them to 0 bytes (`truncate(0)` + `fsync`), which
        Editors ARE permitted to do on files owned by another account! Even if
        `unlink` is denied or the file goes to Trash, it occupies 0 bytes.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

from .model import RetNetConfig, RetNetLM

_STEP_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")
_MIN_VALID_BYTES = 1024  # 0-byte truncated or empty placeholder files are skipped


def _step_from_name(path: Path) -> int:
    m = _STEP_RE.match(path.name)
    return int(m.group(1)) if m else -1


def checkpoint_candidates(run_dir: Path) -> list[Path]:
    """Return candidate checkpoint paths in `run_dir`, newest step first."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    step_files: list[tuple[int, float, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_file() or p.name.endswith(".tmp"):
            continue
        try:
            if p.stat().st_size < _MIN_VALID_BYTES:
                continue
            mtime = p.stat().st_mtime
        except OSError:
            continue
        step = _step_from_name(p)
        if step >= 0:
            step_files.append((step, mtime, p))
    step_files.sort(key=lambda t: (t[0], t[1]), reverse=True)
    ordered = [p for _, _, p in step_files]

    # Backward compatibility: include ckpt.pt if present and valid
    legacy = run_dir / "ckpt.pt"
    try:
        if legacy.is_file() and legacy.stat().st_size >= _MIN_VALID_BYTES and legacy not in ordered:
            ordered.append(legacy)
    except OSError:
        pass
    return ordered


def compact_optimizer_state(opt_state: dict) -> dict:
    """Return a copy of `opt.state_dict()` with fp32 AdamW moment buffers
    (`exp_avg`, `exp_avg_sq`) downcast to bfloat16 on CPU, halving optimizer
    checkpoint size on Google Drive."""
    out = {"param_groups": copy.deepcopy(opt_state["param_groups"]), "state": {}}
    for k, v in opt_state.get("state", {}).items():
        sv = {}
        for sk, st in v.items():
            if isinstance(st, torch.Tensor) and sk in ("exp_avg", "exp_avg_sq") and st.is_floating_point():
                sv[sk] = st.detach().to(device="cpu", dtype=torch.bfloat16)
            elif isinstance(st, torch.Tensor):
                sv[sk] = st.detach().cpu()
            else:
                sv[sk] = copy.deepcopy(st)
        out["state"][k] = sv
    return out


def restore_optimizer_state(opt: torch.optim.Optimizer, opt_state: dict, device: str) -> None:
    """Load optimizer state dict and ensure moment tensors match parameter dtype/device."""
    opt.load_state_dict(opt_state)
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            for sk in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if sk in st and isinstance(st[sk], torch.Tensor):
                    st[sk] = st[sk].to(device=device, dtype=p.dtype)


def _truncate_and_remove(path: Path) -> None:
    """Truncate file to 0 bytes (reclaiming Drive quota even across accounts or
    in Drive Trash), then attempt unlink."""
    try:
        if path.is_file():
            with open(path, "wb") as f:
                f.truncate(0)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
    except OSError:
        pass
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _copy_and_fsync(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` (via same-account `.tmp` when possible) and fsync."""
    tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        with open(tmp, "rb") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, dst)
        return
    except OSError:
        _truncate_and_remove(tmp)
    # Fallback: write in-place (works when editing an existing file owned by another account)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
        fdst.flush()
        try:
            os.fsync(fdst.fileno())
        except OSError:
            pass


def save_checkpoint(
    obj: dict,
    run_dir: Path,
    keep_ckpts: int = 2,
    also_write_legacy_ckpt: bool = False,
) -> Path | None:
    """Stage checkpoint to fast local `/tmp`, sync to `run_dir/ckpt_step_XXXXXXX.pt`,
    update `latest.json`, and prune old checkpoints (truncating to 0 bytes first).

    Never raises on I/O errors so a transient Google Drive hiccup does not
    abort a training run.
    """
    run_dir = Path(run_dir)
    step = int(obj.get("step", 0))
    step_name = f"ckpt_step_{step:07d}.pt"
    target = run_dir / step_name
    fd, local_tmp_str = tempfile.mkstemp(prefix=f"retnet_ckpt_{step}_", suffix=".pt")
    os.close(fd)
    local_tmp = Path(local_tmp_str)

    try:
        torch.save(obj, local_tmp)
        if local_tmp.stat().st_size < _MIN_VALID_BYTES:
            raise OSError("serialized checkpoint unexpectedly small")
        _copy_and_fsync(local_tmp, target)
        if also_write_legacy_ckpt:
            try:
                _copy_and_fsync(local_tmp, run_dir / "ckpt.pt")
            except OSError:
                pass
    except (OSError, RuntimeError) as e:
        print(f"warning: checkpoint save to {target} failed, keeping previous one: {e}", flush=True)
        _truncate_and_remove(local_tmp)
        return None
    finally:
        _truncate_and_remove(local_tmp)

    # Update latest.json metadata pointer (best effort across accounts)
    meta_str = json.dumps(
        {"step": step, "file": step_name, "saved_at": time.time()}, indent=2
    )
    try:
        (run_dir / "latest.json").write_text(meta_str)
    except OSError:
        pass

    # Rotate older step checkpoints (keep newest `keep_ckpts`)
    if keep_ckpts > 0:
        all_step_ckpts = [
            p for p in checkpoint_candidates(run_dir) if _step_from_name(p) >= 0
        ]
        for old_p in all_step_ckpts[keep_ckpts:]:
            if old_p != target:
                _truncate_and_remove(old_p)

    return target


def load_latest_checkpoint(
    run_dir: Path, device: str
) -> tuple[dict | None, Path | None]:
    """Try loading checkpoints in `run_dir` from newest to oldest.

    Returns (checkpoint_dict, path_loaded) or (None, None) if no checkpoint exists.
    Raises SystemExit only if checkpoints exist on disk but ALL of them are corrupted.
    """
    cands = checkpoint_candidates(run_dir)
    if not cands:
        return None, None
    errors: list[str] = []
    for p in cands:
        try:
            ck = torch.load(p, map_location=device, weights_only=False)
            if not isinstance(ck, dict) or "model" not in ck or "config" not in ck:
                raise ValueError("checkpoint missing 'model' or 'config' keys")
            if errors:
                print(
                    f"warning: skipped corrupted newer checkpoint(s) ({'; '.join(errors)}); "
                    f"recovered cleanly from {p.name} (step {ck.get('step', 0)})",
                    flush=True,
                )
            return ck, p
        except Exception as e:
            errors.append(f"{p.name}: {e!r}")
    sys.exit(
        f"error: found checkpoint file(s) in {run_dir} but none could be loaded:\n  "
        + "\n  ".join(errors)
    )


def load_run(run: Path, device: str, data: Path | None = None):
    """Return (model in eval mode, tokenizer, step) from the latest valid checkpoint."""
    from tokenizers import Tokenizer

    run = Path(run)
    tok_path = run / "tokenizer.json"
    if not tok_path.is_file() and data is not None:
        tok_path = Path(data) / "tokenizer.json"
    if not tok_path.is_file():
        sys.exit(f"error: missing tokenizer at {tok_path}")

    ck, ck_path = load_latest_checkpoint(run, device)
    if ck is None or ck_path is None:
        sys.exit(f"error: no checkpoint found in {run}")
    try:
        model = RetNetLM(RetNetConfig(**ck["config"])).to(device).eval()
        model.load_state_dict(ck["model"])
    except Exception as e:
        sys.exit(f"error: could not build model from {ck_path}: {e!r}")
    try:
        tok = Tokenizer.from_file(str(tok_path))
    except Exception as e:
        sys.exit(f"error: could not load tokenizer {tok_path}: {e!r}")
    return model, tok, int(ck.get("step", 0))
