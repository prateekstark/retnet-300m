"""Generate text from a trained checkpoint (recurrent decoding, O(1) per token).

  python scripts/sample.py --run runs/retnet-300m --prompt "Problem:\nWrite a Python function" -n 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.checkpoint import load_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--prompt", default="Question: What is 16 * 47?\nAnswer:")
    ap.add_argument("-n", "--num_samples", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model, tok, step = load_run(a.run, a.device, a.data)
    prompt = a.prompt.replace("\\n", "\n")
    ids = torch.tensor([tok.encode(prompt).ids] * a.num_samples, device=a.device)
    t0 = time.time()
    out = model.generate(ids, a.max_new_tokens, a.temperature, a.top_k, eos_id=0)
    dt = time.time() - t0
    for row in out.tolist():
        if 0 in row:
            row = row[: row.index(0)]
        print(tok.decode(row), "\n" + "-" * 60)
    print(f"(step {step}, {out.shape[1] - ids.shape[1]} new tokens in {dt:.1f}s)")


if __name__ == "__main__":
    main()
