"""prepare_mix: resumable per-source shards, val/train split, stub-source guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_mix as pm  # noqa: E402
from tokenizers import pre_tokenizers, trainers  # noqa: E402


def _rows(n: int, words: int = 40) -> list[dict]:
    return [{"text": " ".join(f"doc{i} word{j}" for j in range(words))} for i in range(n)]


@pytest.fixture
def tok():
    t = pm.make_tokenizer()
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=[pm.EOS], show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t.train_from_iterator([r["text"] for r in _rows(50)], tr)
    return t


def _fake_source(monkeypatch, rows, fail_after=None):
    """Patch iter_rows to serve `rows`; optionally raise after `fail_after` rows (a disconnect)."""
    served = {"n": 0}

    def fake_iter_rows(src, token, skip=0):
        for r in rows[skip:]:
            if fail_after is not None and served["n"] >= fail_after:
                raise ConnectionError("simulated Colab disconnect")
            served["n"] += 1
            yield r

    monkeypatch.setattr(pm, "iter_rows", fake_iter_rows)
    return pm.Source("primary", "fake/ds", None, "train", lambda r: [r["text"]])


def _build(tok, shard_dir, src, budget, val_budget):
    eos = tok.token_to_id(pm.EOS)
    return pm.build_shard("fake", src, tok, eos, budget, val_budget, shard_dir,
                          batch_docs=8, save_every_s=0.0, log_every_s=1e9)


def test_build_shard_resumes_after_disconnect(tmp_path, monkeypatch, tok):
    rows = _rows(300)
    budget, val_budget = 50_000, 4_000

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref = _build(tok, ref_dir, _fake_source(monkeypatch, rows), budget, val_budget)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ConnectionError):
        _build(tok, run_dir, _fake_source(monkeypatch, rows, fail_after=150), budget, val_budget)
    prog = json.loads((run_dir / "fake.progress.json").read_text())
    assert 0 < prog["rows"] <= 150 and prog["n_train"] > 0
    assert not (run_dir / "fake.meta.json").exists()

    out = _build(tok, run_dir, _fake_source(monkeypatch, rows), budget, val_budget)
    assert out == ref
    for split in ("train", "val"):
        a = np.fromfile(ref_dir / f"fake.{split}.bin", dtype=np.uint16)
        b = np.fromfile(run_dir / f"fake.{split}.bin", dtype=np.uint16)
        assert np.array_equal(a, b), split
    assert not (run_dir / "fake.progress.json").exists()
    assert pm.valid_shard(run_dir, "fake", budget, tok.token_to_id(pm.EOS)) is not None


def test_build_shard_multi_pass_skips_val_rows(tmp_path, monkeypatch, tok):
    rows = _rows(40)
    info = _build(tok, tmp_path, _fake_source(monkeypatch, rows), 30_000, 1_500)
    assert info["passes"] > 1
    eos = tok.token_to_id(pm.EOS)
    val = np.fromfile(tmp_path / "fake.val.bin", dtype=np.uint16)
    train = np.fromfile(tmp_path / "fake.train.bin", dtype=np.uint16)
    val_docs = {tok.decode(d.tolist()) for d in np.split(val, np.where(val == eos)[0] + 1) if d.size > 1}
    train_docs = {tok.decode(d.tolist()) for d in np.split(train, np.where(train == eos)[0] + 1) if d.size > 1}
    assert val_docs and train_docs and not (val_docs & train_docs)


def test_stub_source_is_rejected(tmp_path, monkeypatch, tok):
    rows = [{"text": "x"} for _ in range(30_000)]
    with pytest.raises(SystemExit, match="stub rows"):
        _build(tok, tmp_path, _fake_source(monkeypatch, rows), 10**9, 0)


def test_legacy_stub_shard_is_invalidated(tmp_path, tok):
    eos = tok.token_to_id(pm.EOS)
    ids = np.tile(np.array([5, 6, eos], dtype=np.uint16), 1000)  # 2 tokens per doc
    ids.tofile(tmp_path / "old.train.bin")
    ids[:30].tofile(tmp_path / "old.val.bin")
    (tmp_path / "old.meta.json").write_text(json.dumps(
        {"train_tokens": ids.size, "val_tokens": 30, "passes": 1}))
    assert pm.valid_shard(tmp_path, "old", ids.size, eos) is None


def test_resolve_source_falls_back_on_stub_primary(tmp_path, monkeypatch):
    def fake_iter_rows(src, token, skip=0):
        if src.kind == "primary":
            return iter([{"text": "[Not ready yet!]"}] * 16)
        return iter([{"text": "A real textbook paragraph. " * 20}] * 16)

    monkeypatch.setattr(pm, "iter_rows", fake_iter_rows)
    monkeypatch.setattr(pm, "_RESOLVED", {})
    src = pm.resolve_source("tiny_math_textbooks", None, False, tmp_path)
    assert src.kind == "fallback"
    assert json.loads((tmp_path / "tiny_math_textbooks.source.json").read_text())["kind"] == "fallback"
    monkeypatch.setattr(pm, "_RESOLVED", {})
    with pytest.raises(SystemExit):
        pm.resolve_source("tiny_math_textbooks", None, True, tmp_path / "strict")


def test_main_end_to_end_offline(tmp_path, monkeypatch, tok):
    """Full CLI: two sources, shard cache, concatenation, meta.json val ranges."""
    rows = {"gsm8k": [{"question": f"Q{i}? " * 30, "answer": f"{i} + 1 = {i + 1}\n#### {i + 1}"} for i in range(200)],
            "metamath": [{"query": f"M{i}? " * 30, "response": f"{i} * 2 = {2 * i}"} for i in range(200)]}

    def fake_iter_rows(src, token, skip=0):
        key = "gsm8k" if "gsm8k" in src.path else "metamath"
        return iter(rows[key][skip:])

    monkeypatch.setattr(pm, "iter_rows", fake_iter_rows)
    monkeypatch.setattr(pm, "_RESOLVED", {})
    tok_path = tmp_path / "tok.json"
    tok.save(str(tok_path))
    out, cache = tmp_path / "out", tmp_path / "cache"
    argv = ["prepare_mix.py", "--out", str(out), "--cache_dir", str(cache), "--tokenizer", str(tok_path),
            "--total_tokens", "40000", "--val_tokens_per_source", "2000", "--shares", "gsm8k=0.5,metamath=0.5"]
    monkeypatch.setattr(sys, "argv", argv)
    pm.main()
    meta = json.loads((out / "meta.json").read_text())
    assert (out / "train.bin").stat().st_size == 2 * meta["train_tokens"]
    assert (out / "val.bin").stat().st_size == 2 * meta["val_tokens"]
    assert meta["sources"]["gsm8k"]["val_range"][1] == meta["sources"]["metamath"]["val_range"][0]
    # second run reuses cached shards without streaming
    monkeypatch.setattr(pm, "iter_rows", lambda *a, **k: (_ for _ in ()).throw(AssertionError("streamed")))
    pm.main()
    assert json.loads((out / "meta.json").read_text()) == meta
