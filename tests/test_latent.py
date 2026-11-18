"""Latent reasoning: data parsing, batched thought loss vs. unbatched, gradients
through thoughts, generation, and a resumable end-to-end curriculum run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import prepare_mix as pm  # noqa: E402
from retnet.checkpoint import save_checkpoint  # noqa: E402
from retnet.latent import (  # noqa: E402
    BOT,
    EOT,
    LatentReasoner,
    Problem,
    answer_of,
    encode,
    parse_gsm8k,
    parse_metamath,
    special_ids,
)
from retnet.model import RetNetConfig, RetNetLM  # noqa: E402

GSM_ANSWER = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
              "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72")


@pytest.fixture(scope="module")
def tok():
    texts = [f"Question: Tom has {i} apples and buys {i + 3}. How many?\nAnswer: "
             f"{i} + {i + 3} = <<{i}+{i + 3}={2 * i + 3}>>{2 * i + 3}\n#### {2 * i + 3}" for i in range(80)]
    return pm.train_tokenizer(texts, 400)


def _model(tok, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = RetNetConfig(vocab_size=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=2, **kw)
    return RetNetLM(cfg).double().eval()


def _problems(n=6):
    return [Problem(f"Tom has {i} apples and buys {i + 3}. How many?",
                    tuple(f"step {j}: {i} + {j} = <<{i}+{j}={i + j}>>{i + j}" for j in range(1 + i % 3)),
                    str(2 * i + 3)) for i in range(n)]


def test_parsers():
    p = parse_gsm8k("Q?", GSM_ANSWER)
    assert p.answer == "72" and len(p.steps) == 2 and "<<48/2=24>>" in p.steps[0]
    m = parse_metamath("Q?", "She has 3 * 4 = 12 eggs.\nThen 12 + 1 = 13.\nThe answer is: 13")
    assert m.answer == "13" and m.steps[0] == "She has 3 * 4 = <<3*4=12>>12 eggs."
    assert parse_metamath("Q?", "The answer is: \\frac{1}{2}") is None
    assert answer_of(" x\n#### 1,234 dollars") == "1234" and answer_of("no answer") is None


def test_encoding_removes_steps(tok):
    ids = special_ids(tok)
    p = _problems(3)[2]
    e0, e2 = encode(tok, p, 0, ids), encode(tok, p, 2, ids)
    assert e0.prefix == e2.prefix and e0.prefix[-1] == ids[BOT]
    assert e0.suffix[0] == ids[EOT] and e0.suffix[-1] == tok.token_to_id("<|endoftext|>")
    assert "step 0" in tok.decode(e0.suffix) and "step 0" not in tok.decode(e2.suffix)
    assert "#### 7" in tok.decode(e2.suffix)


@pytest.mark.parametrize("short_conv", [0, 4])
def test_zero_thought_loss_matches_plain_forward(tok, short_conv):
    """With no thoughts, the latent path must equal one ordinary forward pass
    over prefix + suffix (checks per-sample prefill, state stacking, offsets)."""
    m = _model(tok, short_conv=short_conv)
    r = LatentReasoner(m)
    ids = special_ids(tok)
    batch = [encode(tok, p, 0, ids) for p in _problems(4)]
    got = r.loss(batch, n_latent=0, chunk_size=8)
    tot, cnt = 0.0, 0
    for b in batch:
        seq = torch.tensor([b.prefix + b.suffix])
        logits, _, _ = m(seq)
        n = len(b.prefix)
        tgt = seq[0, n + 1:]
        tot += F.cross_entropy(logits[0, n: -1], tgt, reduction="sum")
        cnt += tgt.numel()
    assert torch.allclose(got, tot / cnt, atol=1e-9)


def test_thoughts_batched_equal_single(tok):
    m = _model(tok, seed=1)
    r = LatentReasoner(m, scale=0.7)
    ids = special_ids(tok)
    batch = [encode(tok, p, 1, ids) for p in _problems(3)]
    together = r.loss(batch, n_latent=3, chunk_size=8)
    tok_counts = [len(b.suffix) - 1 for b in batch]
    alone = sum(r.loss([b], 3, 8) * n for b, n in zip(batch, tok_counts)) / sum(tok_counts)
    assert torch.allclose(together, alone, atol=1e-9)


def test_gradient_flows_through_thoughts(tok):
    m = _model(tok, seed=2).float().train()
    r = LatentReasoner(m, scale=0.5)
    ids = special_ids(tok)
    loss = r.loss([encode(tok, p, 1, ids) for p in _problems(2)], n_latent=4, chunk_size=8)
    loss.backward()
    assert r.scale.grad is not None and r.scale.grad.abs() > 0
    # the thought path reaches the final norm through hidden states only
    assert m.norm_f.weight.grad.abs().sum() > 0
    r.zero_grad()
    r.loss([encode(tok, p, 1, ids) for p in _problems(2)], n_latent=0, chunk_size=8).backward()
    assert r.scale.grad is None or r.scale.grad.abs() == 0


def test_generate_and_init_scale(tok):
    m = _model(tok, seed=3).float()
    r = LatentReasoner(m)
    s = r.init_scale([encode(tok, p, 0, special_ids(tok)).prefix for p in _problems(4)], 8)
    assert 0 < s < 10 and float(r.scale) == pytest.approx(s)
    outs = r.generate(tok, ["Tom has 2 apples and buys 5. How many?", "Hi?"], n_latent=3,
                      max_new_tokens=12)
    assert len(outs) == 2 and all(isinstance(t, str) for t, _ in outs)


def test_train_latent_end_to_end_resume(tmp_path, monkeypatch, tok):
    """Two-stage curriculum on a tiny model; a run stopped mid-stage resumes and
    finishes with the stage-1 checkpoint carrying the thought settings."""
    import train_latent as tl

    init = tmp_path / "base"
    init.mkdir()
    m = _model(tok, seed=4).float()
    save_checkpoint({"model": m.state_dict(), "config": m.cfg.to_dict(), "step": 7}, init,
                    prune_min_age_s=0)
    tok.save(str(init / "tokenizer.json"))
    probs = _problems(40)
    monkeypatch.setattr(tl, "load_problems", lambda a: (probs, probs[:6]))
    out = tmp_path / "latent"
    base = ["train_latent.py", "--init", str(init), "--out", str(out), "--max_stage", "1",
            "--c", "2", "--epochs_stage0", "1", "--epochs_per_stage", "1", "--batch_size", "4",
            "--grad_accum", "1", "--eval_limit", "6", "--eval_batch", "3", "--log_every", "2",
            "--chunk_size", "8", "--device", "cpu", "--resume"]
    monkeypatch.setattr(sys, "argv", base + ["--max_steps", "3"])
    tl.main()
    monkeypatch.setattr(sys, "argv", base)
    tl.main()
    evals = [json.loads(l) for l in open(out / "log.jsonl") if '"eval"' in l]
    assert [e["stage"] for e in evals] == [0, 1]
    assert evals[1]["eval"]["n_latent"] == 2
    ck = torch.load(sorted(out.glob("ckpt_step_*.pt"))[-1], weights_only=False)
    assert ck["latent"]["stage"] == 1 and ck["latent"]["pos"]["stage"] == 2 and ck["latent"]["c"] == 2 and ck["step"] == 20
