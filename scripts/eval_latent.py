"""Evaluate a latent-reasoning run on GSM8K test, sweeping the number of thoughts.

  python scripts/eval_latent.py --run runs/retnet-300m-latent --n_latent 0,2,4,8 --limit 500

Accuracy against thinking length is the key plot: if the thoughts carry real
computation, accuracy should rise with n_latent up to (and maybe past) the
number the model was trained with.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retnet.checkpoint import load_latest_checkpoint, load_run  # noqa: E402
from retnet.latent import PAUSE, LatentReasoner  # noqa: E402
from train_latent import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n_latent", default="", help="comma list; default: the trained number")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 1319 test problems")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no_calculator", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, tok, step = load_run(a.run, a.device)
    ck, _ = load_latest_checkpoint(a.run, "cpu")
    lat = ck.get("latent", {}) if ck else {}
    del ck
    mode = lat.get("thought_mode", "latent")
    reasoner = LatentReasoner(model, lat.get("scale", 1.0), mode,
                              tok.token_to_id(PAUSE)).to(a.device).eval()
    trained = int(lat.get("n_latent", int(lat.get("stage", 0)) * int(lat.get("c", 0))))
    sweep = [int(x) for x in a.n_latent.split(",") if x.strip()] or [trained]
    print(f"{a.run} step {step}: trained with {trained} thoughts (stage {lat.get('stage')}, "
          f"c {lat.get('c')}, mode {mode}), scale {float(reasoner.scale):.3f}")

    from datasets import load_dataset
    from retnet.latent import parse_gsm8k

    test = [p for r in load_dataset("openai/gsm8k", "main", split="test")
            if (p := parse_gsm8k(r["question"], r["answer"]))]
    results = {"run": str(a.run), "step": step, "trained_n_latent": trained, "thought_mode": mode,
               "calculator": not a.no_calculator, "sweep": []}
    for n in sweep:
        t0 = time.time()
        r = evaluate(reasoner, tok, test, n, a.limit, a.batch, not a.no_calculator)
        r["minutes"] = round((time.time() - t0) / 60, 1)
        print(f"n_latent {n:3d}: acc {r['acc']:.3f} +/- {r['stderr']:.3f} ({r['n']} problems)", flush=True)
        results["sweep"].append(r)
    out = a.out or a.run / ("eval_latent_nocalc.json" if a.no_calculator else "eval_latent.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
