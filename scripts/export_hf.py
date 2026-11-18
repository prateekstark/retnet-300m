"""Export a run's newest checkpoint as a Hugging Face model repo, optionally upload.

  python scripts/export_hf.py --run runs/retnet-300m --out export/retnet-300m
  python scripts/export_hf.py --run runs/retnet-300m --out export/retnet-300m --push USER/retnet-300m --private

Writes model.safetensors (weights only, fp32 by default so continued training
and evals match the original bit for bit; the tied LM head is
not stored twice), config.json, tokenizer.json and a README model card with the
eval results found in the run dir. Load it back anywhere with
`load_run("hf://USER/retnet-300m", device)` or `--init hf://USER/retnet-300m`.
Uploading needs `huggingface-cli login` (your own token) first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.checkpoint import load_run  # noqa: E402

GITHUB = "https://github.com/prateekstark/retnet-300m"


def _results(run: Path) -> list[str]:
    rows = []

    def load(name):
        try:
            return json.loads((run / name).read_text())
        except (OSError, ValueError):
            return {}

    for name, label in (("eval_gsm8k256.json", "GSM8K, calculator in the loop"),
                        ("eval_gsm8k256_nocalc.json", "GSM8K, no calculator")):
        g = load(name).get("gsm8k")
        if g:
            rows.append(f"| {label} | {100 * g['acc']:.1f}% ± {100 * g['stderr']:.1f}% | first {g['n']} test problems |")
    mc = load("eval_mc500.json")
    if "mbpp" in mc:
        m = mc["mbpp"]
        rows.append(f"| MBPP pass@1 | {100 * m['pass_at_1']:.1f}% | {100 * m['syntax_valid']:.0f}% syntactically valid |")
    for k, label in (("arc_easy", "ARC-easy"), ("arc_challenge", "ARC-challenge"), ("hellaswag", "HellaSwag")):
        if k in mc:
            rows.append(f"| {label} (normalized) | {100 * mc[k]['acc_norm']:.1f}% | {mc[k]['n']} questions, chance 25% |")
    return rows


def model_card(model, step: int, run: Path, repo: str | None, dtype: str) -> str:
    cfg = model.cfg
    n = model.num_params(False) / 1e6
    rows = _results(run)
    results = ("| Task | Score | Note |\n|---|---|---|\n" + "\n".join(rows)) if rows else "Not evaluated yet."
    repo_ref = f"hf://{repo}" if repo else "hf://USER/REPO"
    return f"""---
license: mit
library_name: pytorch
tags: [retnet, retention, linear-attention, delta-rule, math, code, calculator]
datasets: [nampdn-ai/tiny-codes, nampdn-ai/tiny-textbooks, meta-math/MetaMathQA, openai/gsm8k, HuggingFaceTB/finemath, ise-uiuc/Magicoder-OSS-Instruct-75K]
---

# RetNet-300M (base, step {step:,})

A {n:.0f}M-parameter **Retentive Network** language model trained from scratch on code, math
and synthetic textbooks, with three additions aimed at multi-step reasoning:

- **Delta-rule memory** with negative eigenvalues (`beta_max = {cfg.beta_max}`): the state can overwrite a
  stored value instead of only adding to it.
- **Short causal convolution** (`K = {cfg.short_conv}`) before the q/k/v projections.
- **Calculator in the loop**: arithmetic is written as `<<48/2=24>>24` and the exact result is filled in
  at decode time.

Inference is recurrent: constant memory per generated token, no KV cache.
Architecture: `d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}`, vocab {cfg.vocab_size:,}
(digit-split BPE, ids 1-16 reserved for post-training).

**Training:** about 494M tokens on a single 16 GB T4 GPU. This is a small research model: it is not
instruction-tuned and will ramble or repeat itself on open-ended prompts.

## Results (zero-shot, greedy)

{results}

## Usage

This is not a `transformers` model; it needs the code from [{GITHUB}]({GITHUB}).

```python
from retnet.checkpoint import load_run
from retnet.calculator import generate_with_calculator

model, tok, step = load_run("{repo_ref}", "cuda")
text, calls = generate_with_calculator(
    model, tok, "Question: A baker bakes 7 trays of 24 muffins and sells 150. How many are left?\\nAnswer:",
    max_new_tokens=200, stop=("\\nQuestion:",))
print(text)
```

## Files

- `model.safetensors`: weights ({dtype}; the LM head is tied to the embedding and not stored)
- `config.json`: model config and training step
- `tokenizer.json`: the tokenizer (Hugging Face `tokenizers` format)

## License

Code and weights: MIT. The training datasets keep their own licenses; check them before commercial use.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir (newest valid checkpoint is exported)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--push", default=None, help="HF repo id to upload to, e.g. USER/retnet-300m")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    from safetensors.torch import save_file

    run = Path(a.run)
    model, tok, step = load_run(run, "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    sd = model.state_dict()
    if model.cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)
    sd = {k: v.detach().to(dtype).contiguous() for k, v in sd.items()}

    a.out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(a.out / "model.safetensors"), metadata={"format": "pt", "step": str(step)})
    (a.out / "config.json").write_text(json.dumps(
        {"architecture": "RetNetLM", "model": model.cfg.to_dict(), "step": step,
         "dtype": a.dtype, "code": GITHUB}, indent=2))
    shutil.copyfile(run / "tokenizer.json", a.out / "tokenizer.json")
    (a.out / "README.md").write_text(model_card(model, step, run, a.push, a.dtype))
    size = (a.out / "model.safetensors").stat().st_size / 2**30
    print(f"exported step {step} to {a.out} ({size:.2f} GiB, {a.dtype})")

    if a.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.push, private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(a.out), repo_id=a.push,
                          commit_message=f"RetNet-300M base, step {step}")
        print(f"uploaded to https://huggingface.co/{a.push} ({'private' if a.private else 'public'})")


if __name__ == "__main__":
    main()
