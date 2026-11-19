"""Measure latent-training throughput for a few settings on this GPU.

  python scripts/bench_latent.py --init hf://starkprateek1/retnet-300m --batches 8,16,32 --n_latent 8

For each (batch size, compile) it runs a few warm-up steps, then times forward +
backward + AdamW steps on GSM8K-Aug problems with the given number of thoughts,
and prints problems/s and peak memory. Pick the fastest setting that fits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.checkpoint import load_run, resolve_run  # noqa: E402
from retnet.latent import LatentReasoner, encode, parse_gsm8k_aug, special_ids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="hf://starkprateek1/retnet-300m")
    ap.add_argument("--batches", default="8,16,32")
    ap.add_argument("--n_latent", type=int, default=8)
    ap.add_argument("--stage", type=int, default=4, help="steps replaced by thoughts")
    ap.add_argument("--compile", default="0,1", help="0, 1 or 0,1")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from datasets import load_dataset

    rows = load_dataset("whynlp/gsm8k-aug", split="train").select(range(4000))
    probs = [p for r in rows if (p := parse_gsm8k_aug(r["question"], r["steps"], r["answer"]))]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"{torch.cuda.get_device_name(0)}, {dtype}, {a.n_latent} thoughts, stage {a.stage}")
    for comp in [int(c) for c in a.compile.split(",")]:
        model, tok, _ = load_run(resolve_run(a.init), a.device)
        ids = special_ids(tok)
        if comp:
            for blk in model.blocks:
                blk.compile(dynamic=True)
        r = LatentReasoner(model, 0.014).to(a.device).train()
        opt = torch.optim.AdamW(r.parameters(), lr=1e-5, fused=True)
        for B in [int(b) for b in a.batches.split(",")]:
            torch.cuda.reset_peak_memory_stats()
            try:
                times = []
                for i in range(a.steps + 3):
                    batch = [encode(tok, p, a.stage, ids) for p in probs[i * B:(i + 1) * B]]
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.autocast("cuda", dtype=dtype):
                        loss = r.loss(batch, a.n_latent, 128)
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    if i >= 3:
                        times.append(time.time() - t0)
                dt = sum(times) / len(times)
                print(f"compile={comp} batch={B:3d}: {B / dt:7.1f} problems/s, {dt * 1000:6.0f} ms/step, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GiB", flush=True)
            except torch.OutOfMemoryError:
                print(f"compile={comp} batch={B:3d}: out of memory", flush=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        del model, r, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
