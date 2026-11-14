"""Correctness tests for RetNet 2.0 (300M presets, parallel/recurrent/chunkwise
mode equivalence, delta-rule memory, short conv, and activation checkpointing).

Run with:  python -m pytest -q tests
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet import MODEL_PRESETS, MultiScaleRetention, RetNetConfig, RetNetLM, config_from_preset

DT = torch.float64  # exact comparisons on CPU


def _close(a, b, atol=1e-9, rtol=1e-7):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


MEMORIES = [
    ("none", 1.0, 0),
    ("delta", 1.0, 0),
    ("delta", 2.0, 0),
    ("none", 1.0, 4),
    ("delta", 2.0, 4),
]


def _layer(d=32, h=4, vf=2, memory="none", beta_max=1.0, short_conv=0):
    return (
        MultiScaleRetention(
            d, h, value_factor=vf, memory=memory, beta_max=beta_max, short_conv=short_conv
        )
        .to(DT)
        .eval()
    )


def _model(**kw):
    cfg = RetNetConfig(vocab_size=50, d_model=32, n_layers=3, n_heads=4, **kw)
    return RetNetLM(cfg).to(DT).eval()


# ---------------------------------------------------------------- layer level
@pytest.mark.parametrize("h,vf", [(1, 1), (4, 2), (8, 1)])
def test_layer_recurrent_matches_parallel(h, vf):
    layer = _layer(h=h, vf=vf)
    x = torch.randn(2, 23, 32, dtype=DT)
    par, _ = layer(x, mode="parallel")
    rec, _ = layer(x, mode="recurrent")
    _close(par, rec)


@pytest.mark.parametrize("memory,beta_max,short_conv", MEMORIES)
@pytest.mark.parametrize("chunk", [1, 3, 7, 16, 23, 64])
def test_layer_chunkwise_matches_parallel(chunk, memory, beta_max, short_conv):
    layer = _layer(memory=memory, beta_max=beta_max, short_conv=short_conv)
    x = torch.randn(2, 23, 32, dtype=DT)
    par, par_state = layer(x, mode="parallel", return_state=True)
    chk, chk_state = layer(x, mode="chunkwise", chunk_size=chunk)
    _, rec_state = layer(x, mode="recurrent")
    _close(par, chk)
    _close(par_state, chk_state)
    _close(par_state, rec_state)


@pytest.mark.parametrize("memory,beta_max,short_conv", MEMORIES)
@pytest.mark.parametrize("split", [1, 5, 12, 22])
@pytest.mark.parametrize("mode2", ["parallel", "recurrent", "chunkwise"])
def test_layer_streaming(split, mode2, memory, beta_max, short_conv):
    """Process a prefix, then continue from its state at the right offset."""
    layer = _layer(memory=memory, beta_max=beta_max, short_conv=short_conv)
    x = torch.randn(2, 23, 32, dtype=DT)
    full, _ = layer(x, mode="parallel")
    a, state = layer(x[:, :split], mode="parallel", return_state=True)
    b, _ = layer(x[:, split:], state=state, offset=split, mode=mode2, chunk_size=4)
    _close(full, torch.cat([a, b], dim=1))


def test_long_sequence_no_overflow():
    """Decay mask must not produce inf/nan at long lengths."""
    layer = MultiScaleRetention(32, 4, memory="none", short_conv=0).eval()
    x = torch.randn(1, 3000, 32)
    out, _ = layer(x, mode="parallel")
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- model level
@pytest.mark.parametrize("memory,beta_max,short_conv", MEMORIES)
def test_model_modes_match(memory, beta_max, short_conv):
    model = _model(memory=memory, beta_max=beta_max, short_conv=short_conv)
    ids = torch.randint(0, 50, (2, 19))
    par, _, _ = model(ids, mode="parallel")
    rec, _, _ = model(ids, mode="recurrent")
    chk, _, _ = model(ids, mode="chunkwise", chunk_size=5)
    _close(par, rec)
    _close(par, chk)


def test_model_is_causal():
    model = _model()
    ids = torch.randint(0, 50, (1, 16))
    base, _, _ = model(ids)
    ids2 = ids.clone()
    ids2[0, 10:] = (ids2[0, 10:] + 1) % 50
    pert, _, _ = model(ids2)
    _close(base[:, :10], pert[:, :10])
    assert not torch.allclose(base[:, 10:], pert[:, 10:])


def test_greedy_generate_matches_parallel_recompute():
    model = _model()
    prompt = torch.randint(0, 50, (2, 7))
    out = model.generate(prompt, max_new_tokens=10, temperature=0, prefill_chunk=3)
    ids = prompt
    for _ in range(10):
        logits, _, _ = model(ids)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, ids)


def test_grad_checkpointing_matches_standard_gradients():
    """Per-block activation checkpointing must produce identical loss & gradients."""
    torch.manual_seed(42)
    cfg = RetNetConfig(vocab_size=40, d_model=32, n_layers=3, n_heads=4, memory="delta", beta_max=2.0, short_conv=4)
    m1 = RetNetLM(cfg).to(DT).train()
    m2 = RetNetLM(cfg).to(DT).train()
    m2.load_state_dict(m1.state_dict())
    m2.set_grad_checkpointing(True)

    ids = torch.randint(0, 40, (2, 17))
    _, loss1, _ = m1(ids[:, :-1], ids[:, 1:], mode="chunkwise", chunk_size=8)
    loss1.backward()
    _, loss2, _ = m2(ids[:, :-1], ids[:, 1:], mode="chunkwise", chunk_size=8)
    loss2.backward()

    _close(loss1, loss2)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        _close(p1.grad, p2.grad)


def test_300m_preset_config_and_param_formula():
    cfg = config_from_preset("300m", vocab_size=32000)
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads) == (1024, 22, 16)
    assert cfg.memory == "delta" and cfg.beta_max == 2.0 and cfg.short_conv == 4
    # Verify exact parameter budget on a scaled-down block
    d, L = 64, 2
    plain = RetNetLM(RetNetConfig(vocab_size=10, d_model=d, n_layers=L, n_heads=4, memory="none", short_conv=0))
    per_layer = sum(p.numel() for p in plain.blocks[0].parameters())
    assert per_layer == 12 * d * d + 4 * d
    assert set(MODEL_PRESETS) == {"300m", "300m-deep", "125m", "30m"}


def test_delta_memory_overwrites():
    """Write v1 then v2 under the same key, read v2."""
    torch.manual_seed(0)
    layer = _layer(memory="delta", short_conv=0)
    H, dk, dv = 4, layer.head_k, layer.head_v
    k = torch.nn.functional.normalize(torch.randn(1, H, 1, dk, dtype=DT), dim=-1).expand(1, H, 2, dk)
    v = torch.randn(1, H, 2, dv, dtype=DT)
    beta = torch.ones(1, H, 2, dtype=DT)
    _, S = layer._recurrent(k, k, v, None, beta=beta)
    read = (k[:, :, :1] @ S).squeeze(2)
    _close(read, v[:, :, 1], atol=1e-9, rtol=1e-9)
    _, S_plain = layer._recurrent(k, k, v, None, beta=None)
    blend = (k[:, :, :1] @ S_plain).squeeze(2)
    assert not torch.allclose(blend, read)
