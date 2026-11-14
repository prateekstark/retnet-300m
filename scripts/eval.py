"""Standard reasoning & coding evaluation suite for a trained RetNet run.

  python scripts/eval.py --run runs/retnet-300m --data data/tinymix   # all tasks
  python scripts/eval.py --run runs/retnet-300m --tasks gsm8k --limit 200
  python scripts/eval.py --run runs/retnet-300m --no_calculator       # ablation

Tasks:
  gsm8k          GSM8K test (1,319): zero-shot "Question: ...\\nAnswer:", greedy,
                 calculator in the loop; exact match on the "#### N" answer.
  arc_easy       ARC-Easy test, multiple choice by log-likelihood (acc, acc_norm)
  arc_challenge  ARC-Challenge test, same
  hellaswag      HellaSwag validation, same
  mbpp           Sanitized MBPP test (Python coding): greedy code generation +
                 syntax compilation check & assertion pass rate (sandboxed subprocess)
  val_by_source  Held-out next-token loss per Tiny-Series / reasoning source
                 (tiny_codes, tiny_math_textbooks, tiny_code_textbooks, etc.)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.calculator import generate_batch_greedy  # noqa: E402
from retnet.checkpoint import load_run  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _load(path, config, split):
    from datasets import load_dataset

    try:
        return load_dataset(path, config, split=split)
    except Exception as e:
        sys.exit(f"error: could not load {path} ({config}/{split}): {e!r}")


def _number(text: str) -> str | None:
    """Final answer: the number after '####', else the last number in the text."""
    part = text.split("####", 1)[1] if "####" in text else text
    nums = NUM.findall(part)
    if not nums:
        return None
    n = (nums[0] if "####" in text else nums[-1]).replace(",", "").rstrip(".")
    try:
        f = float(n)
    except ValueError:
        return None
    return str(int(f)) if f.is_integer() else str(f)


ANSWER_LINE = r"####[^\n]*\d[^\n]*\n"


def eval_gsm8k(model, tok, limit, use_calc, batch=64, show=3):
    """Greedy, zero-shot, all problems decoded `batch` at a time."""
    ds = _load("openai/gsm8k", "main", "test")
    n = min(limit or len(ds), len(ds))
    correct = calls = 0
    for lo in range(0, n, batch):
        rows = [ds[i] for i in range(lo, min(n, lo + batch))]
        outs = generate_batch_greedy(
            model,
            tok,
            [f"Question: {r['question']}\nAnswer:" for r in rows],
            max_new_tokens=320,
            stop=("\nQuestion:", "\n\n"),
            stop_regex=ANSWER_LINE,
            use_calculator=use_calc,
        )
        for j, (r, (text, c)) in enumerate(zip(rows, outs)):
            gold, pred = _number(r["answer"]), _number(text)
            calls += c
            correct += pred is not None and pred == gold
            if lo + j < show:
                print(f"--- gsm8k #{lo + j}: gold {gold}, pred {pred}\n{text.strip()[:400]}\n")
        done = min(n, lo + batch)
        print(f"  gsm8k {done}/{n}: acc {correct / done:.3f}", flush=True)
    se = (correct / n * (1 - correct / n) / n) ** 0.5
    return {"acc": correct / n, "stderr": se, "n": n, "calculator_calls_per_problem": calls / n}


def _extract_python_code(text: str) -> str:
    """Extract Python code from markdown fenced blocks or raw completion."""
    if "```python" in text:
        block = text.split("```python", 1)[1]
        return block.split("```", 1)[0].strip()
    if "```" in text:
        block = text.split("```", 1)[1]
        return block.split("```", 1)[0].strip()
    return text.strip()


def eval_mbpp(model, tok, limit=100, batch=32, run_tests=True):
    """Evaluate Python code generation on MBPP (sanitized test split)."""
    ds = _load("google-research-datasets/mbpp", "sanitized", "test")
    n = min(limit or len(ds), len(ds))
    prompts = []
    rows = [ds[i] for i in range(n)]
    for r in rows:
        sig_hint = ""
        for line in r["code"].splitlines():
            if line.strip().startswith("def "):
                sig_hint = f"\nUse signature: `{line.strip()}`"
                break
        prompts.append(f"Problem:\n{r['prompt']}{sig_hint}\n\nSolution (Python):\n")

    syntax_ok = passed = 0
    for lo in range(0, n, batch):
        sub_prompts = prompts[lo : min(n, lo + batch)]
        sub_rows = rows[lo : min(n, lo + batch)]
        outs = generate_batch_greedy(
            model,
            tok,
            sub_prompts,
            max_new_tokens=256,
            stop=("\nProblem:", "\nQuestion:"),
            use_calculator=False,
        )
        for r, (gen, _) in zip(sub_rows, outs):
            code = _extract_python_code(gen)
            try:
                ast.parse(code)
                syntax_ok += 1
            except SyntaxError:
                continue
            if run_tests:
                test_script = "\n".join(r.get("test_imports", [])) + "\n" + code + "\n" + "\n".join(r["test_list"])
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tf:
                    tf.write(test_script)
                    tf.flush()
                    try:
                        res = subprocess.run(
                            [sys.executable, "-I", tf.name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=3.0,
                        )
                        if res.returncode == 0:
                            passed += 1
                    except subprocess.TimeoutExpired:
                        pass
        done = min(n, lo + batch)
        print(
            f"  mbpp {done}/{n}: syntax_valid {syntax_ok / done:.3f}, pass@1 {passed / done:.3f}",
            flush=True,
        )
    return {"syntax_valid": syntax_ok / n, "pass_at_1": passed / n, "n": n}


@torch.no_grad()
def _choice_logprobs(model, tok, ctx: str, conts: list[str], device):
    """Sum of log p(continuation tokens | context) for each continuation."""
    ctx_ids = tok.encode(ctx).ids
    seqs, spans = [], []
    for c in conts:
        cid = tok.encode(c).ids
        seqs.append(ctx_ids + cid)
        spans.append((len(ctx_ids), len(ctx_ids) + len(cid)))
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    logp = F.log_softmax(model(x.to(device))[0].float(), dim=-1)
    out = []
    for i, (a, b) in enumerate(spans):
        tgt = x[i, a:b].to(device)
        out.append(logp[i, a - 1 : b - 1].gather(-1, tgt[:, None]).sum().item())
    return out


def eval_mc(model, tok, items, device, name):
    acc = acc_norm = 0
    for i, (ctx, conts, label) in enumerate(items):
        lps = _choice_logprobs(model, tok, ctx, conts, device)
        acc += int(max(range(len(lps)), key=lambda j: lps[j]) == label)
        norm = [lp / max(1, len(c.encode())) for lp, c in zip(lps, conts)]
        acc_norm += int(max(range(len(norm)), key=lambda j: norm[j]) == label)
        if (i + 1) % 500 == 0:
            print(f"  {name} {i + 1}/{len(items)}: acc {acc / (i + 1):.3f}", flush=True)
    n = len(items)
    return {"acc": acc / n, "acc_norm": acc_norm / n, "n": n}


def arc_items(config, limit):
    ds = _load("allenai/ai2_arc", config, "test")
    items = []
    for r in ds.select(range(min(limit or len(ds), len(ds)))):
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        items.append(
            (
                f"Question: {r['question']}\nAnswer:",
                [" " + t for t in r["choices"]["text"]],
                labels.index(r["answerKey"]),
            )
        )
    return items


def hellaswag_items(limit):
    ds = _load("Rowan/hellaswag", None, "validation")

    def clean(t):
        return re.sub(r"\[.*?\]", "", t.replace(" [title]", ". ")).replace("  ", " ").strip()

    items = []
    for r in ds.select(range(min(limit or len(ds), len(ds)))):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        items.append(
            (
                clean(r["activity_label"] + ": " + ctx),
                [" " + clean(e) for e in r["endings"]],
                int(r["label"]),
            )
        )
    return items


@torch.no_grad()
def eval_val_by_source(model, data: Path | None, device, seq_len=1024, max_tokens=262144, bs=4):
    """Mean next-token loss on each source's held-out slice of val.bin."""
    import numpy as np

    if data is None:
        sys.exit("error: val_by_source needs --data (the folder with val.bin and meta.json)")
    try:
        meta = json.loads((data / "meta.json").read_text())
        val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")
    except (OSError, ValueError) as e:
        sys.exit(f"error: could not read {data}/meta.json or val.bin: {e!r}")
    out = {}
    for name, info in meta.get("sources", {}).items():
        if "val_range" not in info:
            continue
        lo, hi = info["val_range"]
        hi = min(hi, lo + max_tokens)
        starts = list(range(lo, hi - seq_len - 1, seq_len))
        if not starts:
            continue
        losses = []
        for i in range(0, len(starts), bs):
            chunk = np.stack([val[s : s + seq_len + 1] for s in starts[i : i + bs]]).astype(np.int64)
            t = torch.from_numpy(chunk).to(device)
            _, loss, _ = model(t[:, :-1], t[:, 1:], mode="chunkwise", chunk_size=128)
            losses.append(loss.item())
        out[name] = round(float(np.mean(losses)), 4)
        print(
            f"  {name:22s} val loss {out[name]:.4f}  ({len(starts) * seq_len / 1e3:.0f}k tokens)",
            flush=True,
        )
    if not out:
        print("  (no per-source ranges in meta.json)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--tasks", default="gsm8k,arc_easy,arc_challenge,hellaswag,mbpp")
    ap.add_argument("--limit", type=int, default=0, help="max examples per task (0 = all)")
    ap.add_argument("--no_calculator", action="store_true")
    ap.add_argument("--gen_batch", type=int, default=64, help="GSM8K problems decoded together")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results file (default <run>/eval.json, or eval_nocalc.json)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, tok, step = load_run(a.run, a.device, a.data)
    results = {
        "run": str(a.run),
        "step": step,
        "memory": model.cfg.memory,
        "beta_max": model.cfg.beta_max,
        "short_conv": model.cfg.short_conv,
        "params_M": round(model.num_params(False) / 1e6, 1),
        "calculator": not a.no_calculator,
        "limit": a.limit,
    }
    for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        t0 = time.time()
        print(f"== {task}", flush=True)
        if task == "gsm8k":
            r = eval_gsm8k(model, tok, a.limit, not a.no_calculator, a.gen_batch)
        elif task in ("arc_easy", "arc_challenge"):
            cfg = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
            r = eval_mc(model, tok, arc_items(cfg, a.limit), a.device, task)
        elif task == "hellaswag":
            r = eval_mc(model, tok, hellaswag_items(a.limit), a.device, task)
        elif task == "mbpp":
            r = eval_mbpp(model, tok, a.limit or 100, batch=min(32, a.gen_batch))
        elif task == "val_by_source":
            r = eval_val_by_source(model, a.data, a.device)
        else:
            sys.exit(f"error: unknown task {task!r}")
        r["minutes"] = round((time.time() - t0) / 60, 1)
        results[task] = r
        print(f"   {task}: {json.dumps(r)}", flush=True)

    print("\n" + json.dumps(results, indent=2))
    out = a.out or a.run / ("eval_nocalc.json" if a.no_calculator else "eval.json")
    try:
        out.write_text(json.dumps(results, indent=2))
    except OSError as e:
        print(f"warning: could not write eval results: {e}")


if __name__ == "__main__":
    main()
