"""Multi-account Google Drive shared folder discovery and verified data caching.

How to share a folder across multiple Google Colab accounts (Personal Gmail):
  1. Account 1 (Owner):
     - In Google Drive, create a folder named `retnet-300m` in `My Drive`.
     - Right-click `retnet-300m` -> Share -> Add Account 2, Account 3, ... as **Editor**.
  2. Account 2, 3, ... (Collaborators):
     - Open Google Drive -> click **Shared with me** in the left sidebar.
     - Right-click the shared `retnet-300m` folder -> **Organize** -> **Add shortcut**.
     - Choose **My Drive** and click **Add**.
     (Colab's `drive.mount('/content/drive')` does NOT mount "Shared with me"
     directly; adding a shortcut into `My Drive` makes it appear at
     `/content/drive/MyDrive/retnet-300m` on every account!)

This module automatically locates `retnet-300m` whether it is:
  - A direct folder or shortcut at `/content/drive/MyDrive/<folder_name>`
  - A Google Workspace Shared Drive at `/content/drive/Shareddrives/*/<folder_name>`
  - A shortcut target at `/content/drive/.shortcut-targets-by-id/*/<folder_name>`
and verifies both read/write Editor permissions and exact binary file sizes.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def resolve_shared_drive_dir(
    folder_name: str = "retnet-300m",
    mount_root: Path = Path("/content/drive"),
    create_if_missing: bool = False,
) -> Path:
    """Locate the shared `folder_name` across MyDrive, Shareddrives, or
    `.shortcut-targets-by-id`, and verify Editor write access.
    """
    env_override = os.environ.get("RETNET_DRIVE_DIR")
    if env_override:
        p = Path(env_override)
        p.mkdir(parents=True, exist_ok=True)
        _verify_write_access(p)
        return p

    mount_root = Path(mount_root)
    candidates: list[Path] = [mount_root / "MyDrive" / folder_name]

    # Check Google Workspace Shared Drives
    shared_drives = mount_root / "Shareddrives"
    if shared_drives.is_dir():
        for sd in sorted(shared_drives.iterdir()):
            if sd.is_dir():
                candidates.append(sd / folder_name)

    # Check Drive shortcut targets by ID
    shortcut_root = mount_root / ".shortcut-targets-by-id"
    if shortcut_root.is_dir():
        for target_id in sorted(shortcut_root.iterdir()):
            if target_id.is_dir():
                candidates.append(target_id / folder_name)

    for cand in candidates:
        if cand.is_dir():
            _verify_write_access(cand)
            return cand

    if create_if_missing and (mount_root / "MyDrive").is_dir():
        default_dir = mount_root / "MyDrive" / folder_name
        default_dir.mkdir(parents=True, exist_ok=True)
        _verify_write_access(default_dir)
        return default_dir

    raise FileNotFoundError(
        f"Could not find shared folder '{folder_name}' under {mount_root}.\n"
        "Multi-account checklist:\n"
        f"  1. On Owner account: create 'My Drive/{folder_name}' and Share (as Editor) with your other accounts.\n"
        f"  2. On Collaborator accounts: in Google Drive -> 'Shared with me' -> right-click '{folder_name}' "
        "-> Organize -> Add shortcut -> My Drive."
    )


def _verify_write_access(folder: Path) -> None:
    """Ensure the current Google account has Editor (write) permission in `folder`."""
    probe = folder / f".write_probe_{os.getpid()}"
    try:
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
    except OSError as e:
        raise PermissionError(
            f"Folder {folder} exists, but this Google account cannot write to it ({e}).\n"
            "Make sure the folder is shared with 'Editor' (not 'Viewer') permission."
        ) from e


def is_valid_token_cache(data_dir: Path) -> bool:
    """Verify `data_dir` has `meta.json`, `tokenizer.json`, `train.bin`, and `val.bin`
    AND that the `.bin` byte sizes match `meta.json` token counts (`N * 2` bytes for uint16).

    Prevents a half-copied `train.bin` from an interrupted Colab session from
    being mistaken for a complete dataset cache.
    """
    data_dir = Path(data_dir)
    req = ("meta.json", "tokenizer.json", "train.bin", "val.bin")
    if not all((data_dir / f).is_file() for f in req):
        return False
    try:
        meta = json.loads((data_dir / "meta.json").read_text())
        n_train = int(meta["train_tokens"])
        n_val = int(meta["val_tokens"])
        if n_train <= 0 or n_val <= 0:
            return False
        train_bytes = (data_dir / "train.bin").stat().st_size
        val_bytes = (data_dir / "val.bin").stat().st_size
        return train_bytes == n_train * 2 and val_bytes == n_val * 2
    except Exception:
        return False


def sync_data_to_drive(local_dir: Path, drive_dir: Path) -> None:
    """Safely copy prepared dataset from `local_dir` to `drive_dir`.

    Copies `tokenizer.json`, `val.bin`, and `train.bin` first, and writes
    `meta.json` LAST only after verifying byte sizes on Drive.
    """
    local_dir, drive_dir = Path(local_dir), Path(drive_dir)
    if not is_valid_token_cache(local_dir):
        raise ValueError(f"local dataset at {local_dir} is incomplete or invalid")
    drive_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale meta.json during transfer so an interrupted copy isn't marked valid
    drive_meta = drive_dir / "meta.json"
    try:
        drive_meta.unlink(missing_ok=True)
    except OSError:
        pass

    for fname in ("tokenizer.json", "val.bin", "train.bin"):
        src = local_dir / fname
        dst = drive_dir / fname
        tmp = drive_dir / f"{fname}.{os.getpid()}.tmp"
        try:
            shutil.copyfile(src, tmp)
            with open(tmp, "rb") as f:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, dst)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            shutil.copyfile(src, dst)

    shutil.copyfile(local_dir / "meta.json", drive_meta)
    if not is_valid_token_cache(drive_dir):
        raise RuntimeError(f"Drive dataset verification failed after copy to {drive_dir}")


def restore_data_from_drive(drive_dir: Path, local_dir: Path) -> bool:
    """If `drive_dir` contains a verified token cache, copy it to fast local SSD `local_dir`."""
    drive_dir, local_dir = Path(drive_dir), Path(local_dir)
    if not is_valid_token_cache(drive_dir):
        return False
    if is_valid_token_cache(local_dir):
        return True
    local_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "val.bin", "train.bin", "meta.json"):
        shutil.copyfile(drive_dir / fname, local_dir / fname)
    return is_valid_token_cache(local_dir)
