"""Tests for multi-account Shared Drive checkpointing, corrupted-checkpoint
fallback, compact optimizer states, Drive cache verification, and Tiny-Series
data extractors."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from colab_drive import (  # noqa: E402
    is_valid_token_cache,
    resolve_shared_drive_dir,
    restore_data_from_drive,
    sync_data_to_drive,
)
from prepare_mix import _extract_textbook, _extract_tiny_codes  # noqa: E402
from retnet import RetNetConfig, RetNetLM  # noqa: E402
from retnet.checkpoint import (  # noqa: E402
    checkpoint_candidates,
    compact_optimizer_state,
    load_latest_checkpoint,
    restore_optimizer_state,
    save_checkpoint,
)


def _dummy_state(step: int):
    cfg = RetNetConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2)
    model = RetNetLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Run 1 step so AdamW populates exp_avg / exp_avg_sq state tensors
    x = torch.randint(0, 32, (2, 8))
    _, loss, _ = model(x, x)
    loss.backward()
    opt.step()
    return model, opt, cfg, {
        "model": model.state_dict(),
        "optimizer": compact_optimizer_state(opt.state_dict()),
        "scaler": {},
        "step": step,
        "config": cfg.to_dict(),
    }


def test_versioned_checkpoints_and_corrupted_fallback(tmp_path: Path):
    """If Account A disconnects mid-save on step 200, Account B must automatically
    fall back to step 100 without crashing."""
    _, _, _, obj100 = _dummy_state(100)
    _, _, _, obj200 = _dummy_state(200)
    _, _, _, obj300 = _dummy_state(300)

    p100 = save_checkpoint(obj100, tmp_path, keep_ckpts=2)
    p200 = save_checkpoint(obj200, tmp_path, keep_ckpts=2)
    p300 = save_checkpoint(obj300, tmp_path, keep_ckpts=2)
    assert p100 is not None and p200 is not None and p300 is not None

    # Step 100 was pruned (keep_ckpts=2); even if unlink were blocked on Shared Drive,
    # its size would be truncated to 0 bytes and excluded from candidates.
    cands = checkpoint_candidates(tmp_path)
    assert [c.name for c in cands] == ["ckpt_step_0000300.pt", "ckpt_step_0000200.pt"]

    # Simulate a half-written corrupted checkpoint at step 300 from a Colab disconnect
    p300.write_bytes(b"corrupted_half_written_drive_sync" * 100)
    ck, loaded_path = load_latest_checkpoint(tmp_path, "cpu")
    assert ck is not None and loaded_path == p200
    assert ck["step"] == 200


def test_compact_optimizer_roundtrip():
    model, opt, cfg, obj = _dummy_state(50)
    for v in obj["optimizer"]["state"].values():
        assert v["exp_avg"].dtype == torch.bfloat16
        assert v["exp_avg_sq"].dtype == torch.bfloat16

    model2 = RetNetLM(cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    restore_optimizer_state(opt2, obj["optimizer"], "cpu")
    for p in model2.parameters():
        st = opt2.state.get(p)
        if st:
            assert st["exp_avg"].dtype == p.dtype


def test_shared_drive_resolver_finds_shortcut_and_shared_drives(tmp_path: Path):
    # 1. Shortcut target under .shortcut-targets-by-id
    sc = tmp_path / ".shortcut-targets-by-id" / "1AbCdEfG" / "retnet-300m"
    sc.mkdir(parents=True)
    assert resolve_shared_drive_dir("retnet-300m", mount_root=tmp_path) == sc

    # 2. Direct MyDrive path takes precedence when present
    md = tmp_path / "MyDrive" / "retnet-300m"
    md.mkdir(parents=True)
    assert resolve_shared_drive_dir("retnet-300m", mount_root=tmp_path) == md


def test_drive_token_cache_rejects_partial_copy(tmp_path: Path):
    local = tmp_path / "local"
    drive = tmp_path / "drive"
    local.mkdir()
    (local / "tokenizer.json").write_text("{}")
    np.arange(100, dtype=np.uint16).tofile(local / "train.bin")
    np.arange(20, dtype=np.uint16).tofile(local / "val.bin")
    (local / "meta.json").write_text(json.dumps({"train_tokens": 100, "val_tokens": 20}))

    assert is_valid_token_cache(local)
    sync_data_to_drive(local, drive)
    assert is_valid_token_cache(drive)

    # Corrupt/truncate train.bin on Drive (simulating disconnect during copy)
    with open(drive / "train.bin", "ab") as f:
        f.truncate(50)  # only 25 tokens instead of 100
    assert not is_valid_token_cache(drive)
    assert not restore_data_from_drive(drive, tmp_path / "restored")


def test_tiny_series_extractors():
    # tiny-textbooks: `textbook` has full synthesized lesson, `text` has shorter seed snippet
    row_tb = {
        "text": "short web snippet",
        "textbook": "Chapter 1: Recursion is a method where 2 + 3 = 5 in base 10, allowing functions to call themselves.",
    }
    assert _extract_textbook(row_tb, annotate_math=True) == [
        "Chapter 1: Recursion is a method where 2 + 3 = <<2+3=5>>5 in base 10, allowing functions to call themselves."
    ]
    # Short/stub rows (< 60 chars, e.g. bare topic names in draft repos) are filtered out
    assert _extract_textbook({"text": "Algebra", "topic": "Math"}) == []
    # Unknown column names with substantive textbook content are automatically discovered
    row_custom = {
        "topic": "Linear Algebra",
        "markdown_lesson": "Eigenvalues and eigenvectors satisfy A v = lambda v, where 4 * 5 = 20 is a scalar product example.",
    }
    assert _extract_textbook(row_custom, annotate_math=True) == [
        "Eigenvalues and eigenvectors satisfy A v = lambda v, where 4 * 5 = <<4*5=20>>20 is a scalar product example."
    ]
    # tiny-codes: prompt + response + programming_language
    row_code = {
        "prompt": "Write a function to add two numbers.",
        "response": "def add(a, b):\n    return a + b",
        "programming_language": "Python",
    }
    assert _extract_tiny_codes(row_code) == [
        "Problem:\nWrite a function to add two numbers.\n\nSolution (Python):\ndef add(a, b):\n    return a + b"
    ]


def test_end_to_end_multi_account_train_resume_and_chat(tmp_path: Path):
    """Simulate Account A training to step 2, Account B resuming to step 4 even
    if a corrupted step-3 file exists, and chat.py generating with calculator."""
    import subprocess
    from prepare_mix import EOS, make_tokenizer
    from tokenizers import pre_tokenizers, trainers

    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()

    tok = make_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=[EOS],
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(["Question: 48/2 = <<48/2=24>>24\nAnswer: 24"] * 40, trainer)
    tok.save(str(data_dir / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    rng = np.random.default_rng(0)
    train_ids = rng.integers(0, vocab_size, size=4096, dtype=np.uint16)
    val_ids = rng.integers(0, vocab_size, size=1024, dtype=np.uint16)
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")
    (data_dir / "meta.json").write_text(
        json.dumps({"vocab_size": vocab_size, "eos_id": 0, "train_tokens": 4096, "val_tokens": 1024})
    )

    base_cmd = [
        sys.executable,
        "-I",
        str(ROOT / "scripts" / "train.py"),
        "--name",
        "test-run",
        "--data",
        str(data_dir),
        "--out_root",
        str(runs_dir),
        "--d_model",
        "32",
        "--n_layers",
        "2",
        "--n_heads",
        "4",
        "--seq_len",
        "16",
        "--batch_size",
        "2",
        "--grad_accum",
        "1",
        "--eval_batches",
        "1",
        "--log_every",
        "1",
        "--eval_every",
        "2",
        "--save_every",
        "1",
        "--device",
        "cpu",
        "--dtype",
        "fp32",
    ]

    # Account A: train to step 2
    res_a = subprocess.run(base_cmd + ["--max_steps", "2", "--resume"], capture_output=True, text=True)
    assert res_a.returncode == 0, res_a.stdout + res_a.stderr
    run_out = runs_dir / "test-run"
    assert (run_out / "ckpt_step_0000002.pt").is_file()

    # Simulate an interrupted/corrupted step 3 checkpoint left on Drive when Account A disconnected
    (run_out / "ckpt_step_0000003.pt").write_bytes(b"truncated_by_disconnect" * 100)

    # Account B: resume with --max_steps 4; must skip corrupted step 3, load step 2, and finish step 4
    res_b = subprocess.run(base_cmd + ["--max_steps", "4", "--resume"], capture_output=True, text=True)
    assert res_b.returncode == 0, res_b.stdout + res_b.stderr
    assert "recovered cleanly from ckpt_step_0000002.pt" in res_b.stdout
    assert "resumed from ckpt_step_0000002.pt at step 2" in res_b.stdout
    assert (run_out / "ckpt_step_0000004.pt").is_file()

    # Run chat.py one-shot coding prompt on the checkpoint
    res_chat = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts" / "chat.py"),
            "--run",
            str(run_out),
            "--code",
            "def add(a, b):",
            "--max_new_tokens",
            "8",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert res_chat.returncode == 0, res_chat.stdout + res_chat.stderr
    assert "step 4" in res_chat.stdout

