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
  3. Pruning must never destroy the only good copy:
     Drive FUSE `fsync` returns before the upload finishes, and when the Drive
     quota is full a multi-GB upload fails silently while small writes still
     sync. Old checkpoints used to be truncated to 0 bytes before deletion;
     those truncations reached Drive while the new checkpoint never did, which
     destroyed every usable checkpoint (and each truncated file still kept its
     2.5 GB old revision, filling the quota further).
     -> Solution: old checkpoints are only unlinked (recoverable from Trash),
        never truncated, and only once an older full-size checkpoint that is
        at least `prune_min_age_s` old remains as a safe anchor. Low free
        space on the run directory is reported loudly.
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
    """Truncate then unlink. Only for local scratch files: never use it on a
    checkpoint in the run directory (see module docstring, item 3)."""
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


def _remove(path: Path) -> None:
    """Unlink without truncating, so the file stays recoverable from Drive Trash."""
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        print(f"warning: could not remove old checkpoint {path.name}: {e}", flush=True)


def _warn_if_low_space(run_dir: Path, need_bytes: int) -> None:
    try:
        free = shutil.disk_usage(run_dir).free
    except OSError:
        return
    if free < 2 * need_bytes:
        print(
            f"WARNING: only {free / 2**30:.1f} GiB free in {run_dir} "
            f"(checkpoint is {need_bytes / 2**30:.1f} GiB). Uploads may fail silently; "
            "free space (and empty Drive Trash) now.",
            flush=True,
        )


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
    prune_min_age_s: float = 1200.0,
) -> Path | None:
    """Stage checkpoint to fast local `/tmp`, sync to `run_dir/ckpt_step_XXXXXXX.pt`,
    update `latest.json`, and prune old checkpoints.

    Pruning keeps the newest `keep_ckpts` files plus the newest full-size file
    older than `prune_min_age_s` (an anchor that has had time to upload), and
    deletes nothing if no such anchor exists.

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
        size = local_tmp.stat().st_size
        if size < _MIN_VALID_BYTES:
            raise OSError("serialized checkpoint unexpectedly small")
        _warn_if_low_space(run_dir, size)
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

    # Rotate older step checkpoints: keep the newest `keep_ckpts` plus an anchor
    # old enough to have finished uploading; delete nothing without an anchor.
    if keep_ckpts > 0:
        cands = [p for p in checkpoint_candidates(run_dir) if _step_from_name(p) >= 0]
        now = time.time()
        anchor = None
        for p in cands:
            try:
                st = p.stat()
            except OSError:
                continue
            if p != target and now - st.st_mtime >= prune_min_age_s and st.st_size >= 0.9 * size:
                anchor = p
                break
        if anchor is not None:
            keep = set(cands[:keep_ckpts]) | {anchor, target}
            for old_p in cands:
                if old_p not in keep and _step_from_name(old_p) < _step_from_name(anchor):
                    _remove(old_p)

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


def resolve_run(run) -> Path:
    """A local run dir, or `hf://user/repo[@revision]` downloaded from the
    Hugging Face Hub (cached; needs `huggingface_hub`)."""
    s = str(run)
    for prefix in ("hf://", "hf:/"):
        if s.startswith(prefix):
            from huggingface_hub import snapshot_download

            repo, _, rev = s[len(prefix):].partition("@")
            return Path(snapshot_download(repo, revision=rev or None))
    return Path(run)


def load_exported(run: Path, device: str):
    """(model, step) from an exported `model.safetensors` + `config.json`."""
    from safetensors.torch import load_file

    meta = json.loads((run / "config.json").read_text())
    model = RetNetLM(RetNetConfig(**meta["model"]))
    sd = {k: v.float() for k, v in load_file(str(run / "model.safetensors")).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or [k for k in missing if k != "lm_head.weight"]:
        raise RuntimeError(f"weights do not match config: missing {missing}, unexpected {unexpected}")
    return model.to(device).eval(), int(meta.get("step", 0))


def load_run(run, device: str, data: Path | None = None):
    """Return (model in eval mode, tokenizer, step) from the latest valid
    checkpoint, an exported `model.safetensors`, or `hf://user/repo`."""
    from tokenizers import Tokenizer

    run = resolve_run(run)
    tok_path = run / "tokenizer.json"
    if not tok_path.is_file() and data is not None:
        tok_path = Path(data) / "tokenizer.json"
    if not tok_path.is_file():
        sys.exit(f"error: missing tokenizer at {tok_path}")

    if not checkpoint_candidates(run) and (run / "model.safetensors").is_file():
        try:
            model, step = load_exported(run, device)
        except Exception as e:
            sys.exit(f"error: could not load {run / 'model.safetensors'}: {e!r}")
    else:
        ck, ck_path = load_latest_checkpoint(run, device)
        if ck is None or ck_path is None:
            sys.exit(f"error: no checkpoint found in {run}")
        try:
            model = RetNetLM(RetNetConfig(**ck["config"])).to(device).eval()
            model.load_state_dict(ck["model"])
        except Exception as e:
            sys.exit(f"error: could not build model from {ck_path}: {e!r}")
        step = int(ck.get("step", 0))
    try:
        tok = Tokenizer.from_file(str(tok_path))
    except Exception as e:
        sys.exit(f"error: could not load tokenizer {tok_path}: {e!r}")
    return model, tok, step
