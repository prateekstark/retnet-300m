"""Synthetic reasoning probes: each task isolates one ability.

  vartrack : variable tracking with reassignment (memory: last write must win)
             "a 3 b 7 a 5 ... ? a -> 5"
  parity   : running parity of a bit string (state tracking); trained on short
             strings, tested on 4x longer ones (length generalization)

  python scripts/synthetic.py --task vartrack --memory none
  python scripts/synthetic.py --task vartrack --memory delta --short_conv 4
  python scripts/synthetic.py --task parity --memory delta --beta_max 2 --short_conv 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet import RetNetConfig, RetNetLM  # noqa: E402

IGNORE = -1


class VarTrack:
    """[var val] x n_assign, then [? var val] x n_query; loss on the query vals."""

    def __init__(self, n_vars=16, n_vals=32, n_assign=64, n_query=16):
        self.n_vars, self.n_vals, self.n_assign, self.n_query = n_vars, n_vals, n_assign, n_query
        self.Q = 1
        self.var0 = 2
        self.val0 = 2 + n_vars
        self.vocab = self.val0 + n_vals

    def batch(self, bs, device, length_mult=1):
        g = torch.Generator(device="cpu")
        g.seed()
        na = self.n_assign * length_mult
        vars_ = torch.randint(0, self.n_vars, (bs, na), generator=g)
        vals = torch.randint(0, self.n_vals, (bs, na), generator=g)
        last = torch.full((bs, self.n_vars), -1, dtype=torch.long)
        for t in range(na):
            last[torch.arange(bs), vars_[:, t]] = vals[:, t]
        qv = torch.empty(bs, self.n_query, dtype=torch.long)
        for b in range(bs):
            assigned = (last[b] >= 0).nonzero().squeeze(1)
            qv[b] = assigned[torch.randint(0, len(assigned), (self.n_query,), generator=g)]
        qa = last.gather(1, qv)

        assign = torch.stack([vars_ + self.var0, vals + self.val0], dim=2).flatten(1)
        query = torch.stack(
            [torch.full_like(qv, self.Q), qv + self.var0, qa + self.val0], dim=2
        ).flatten(1)
        seq = torch.cat([assign, query], dim=1)
        x, y = seq[:, :-1].clone(), seq[:, 1:].clone()
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, 2 * na + 1 :: 3] = True
        y[~mask] = IGNORE
        return x.to(device), y.to(device)


class Parity:
    """Bits b_1..b_T; target at t is parity(b_1..b_t). Train len T, eval 4T."""

    def __init__(self, length=64):
        self.length = length
        self.vocab = 4

    def batch(self, bs, device, length_mult=1):
        T = self.length * length_mult
        bits = torch.randint(0, 2, (bs, T))
        par = bits.cumsum(1) % 2
        return bits.to(device), (par + 2).to(device)


TASKS = {"vartrack": VarTrack, "parity": Parity}


@torch.no_grad()
def accuracy(model, task, device, n_batches=20, bs=64, length_mult=1):
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        x, y = task.batch(bs, device, length_mult)
        logits, _, _ = model(x)
        m = y != IGNORE
        correct += (logits.argmax(-1)[m] == y[m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="vartrack")
    ap.add_argument("--memory", choices=["none", "delta"], default="delta")
    ap.add_argument("--beta_max", type=float, default=2.0)
    ap.add_argument("--short_conv", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/synthetic/results.jsonl"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    task = TASKS[a.task]()
    cfg = RetNetConfig(
        vocab_size=task.vocab,
        d_model=a.d_model,
        n_layers=a.n_layers,
        n_heads=a.n_heads,
        memory=a.memory,
        beta_max=a.beta_max,
        short_conv=a.short_conv,
        tie_embeddings=False,
    )
    model = RetNetLM(cfg).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))),
    )
    name = f"{a.task} memory={a.memory} beta_max={a.beta_max} short_conv={a.short_conv}"
    print(f"{name}: {model.num_params(False) / 1e6:.2f}M params", flush=True)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, y = task.batch(a.batch_size, a.device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % (a.steps // 6) == 0:
            print(
                f"  step {step:5d} loss {loss.item():.4f} acc {accuracy(model, task, a.device, 4):.3f}",
                flush=True,
            )

    res = {
        "task": a.task,
        "memory": a.memory,
        "beta_max": a.beta_max,
        "short_conv": a.short_conv,
        "seed": a.seed,
        "params": model.num_params(False),
        "acc": accuracy(model, task, a.device),
        "acc_4x_length": accuracy(model, task, a.device, length_mult=4),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(json.dumps(res), flush=True)
    try:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    except OSError as e:
        print(f"warning: could not write {a.out}: {e}")


if __name__ == "__main__":
    main()
