"""Teach a pretrained RetNet to reason in latent space (Coconut curriculum).

Stage 0 is plain step-by-step SFT on GSM8K-style problems. At stage s the
first s reasoning steps are replaced by s * c continuous thoughts (the model's
own hidden state fed back as input, see retnet/latent.py). The optimizer is
reset at every stage, as in Coconut. After each stage the model is evaluated
on GSM8K test with that stage's number of thoughts.

  python scripts/train_latent.py --init runs/retnet-300m --out runs/retnet-300m-latent \\
      --max_stage 4 --c 2 --metamath 40000 --resume

Resumable at any batch: checkpoints go to --out every --save_every_minutes and
at the end of every stage (same rotation as pretraining, retnet/checkpoint.py).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.checkpoint import load_latest_checkpoint, load_run, resolve_run, save_checkpoint  # noqa: E402
from retnet.latent import (  # noqa: E402
    LatentReasoner,
    encode,
    normalize_answer,
    answer_of,
    parse_gsm8k,
    parse_gsm8k_aug,
    parse_metamath,
    PAUSE,
    THOUGHT_MODES,
    special_ids,
)


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name != "auto":
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]
    if not device.startswith("cuda"):
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16 if major >= 7 else torch.float32


def load_problems(a) -> tuple[list, list]:
    """(train problems, GSM8K test problems)."""
    from datasets import load_dataset

    test = [p for r in load_dataset("openai/gsm8k", "main", split="test")
            if (p := parse_gsm8k(r["question"], r["answer"]))]
    if a.data == "gsm8k_aug":
        test_q = {p.question for p in test}
        rows = load_dataset("whynlp/gsm8k-aug", split="train")
        train = [p for r in rows if r["question"].strip() not in test_q
                 and (p := parse_gsm8k_aug(r["question"], r["steps"], r["answer"]))]
        if a.aug_limit:
            random.Random(a.seed).shuffle(train)
            train = train[: a.aug_limit]
        print(f"train problems: {len(train)} GSM8K-Aug (of {len(rows)}; equation steps), "
              f"test: {len(test)}", flush=True)
        return train, test
    train = [p for r in load_dataset("openai/gsm8k", "main", split="train")
             if (p := parse_gsm8k(r["question"], r["answer"]))]
    n_gsm = len(train)
    if a.metamath > 0:
        rng = random.Random(a.seed)
        rows = [r for r in load_dataset("meta-math/MetaMathQA", split="train")
                if str(r.get("type", "")).startswith("GSM")]
        rng.shuffle(rows)
        test_q = {p.question for p in train}
        extra = []
        for r in rows:
            p = parse_metamath(r["query"], r["response"])
            if p and p.question not in test_q:
                extra.append(p)
            if len(extra) >= a.metamath:
                break
        train += extra
    print(f"train problems: {len(train)} ({n_gsm} GSM8K + {len(train) - n_gsm} MetaMathQA-GSM), "
          f"test: {len(test)}", flush=True)
    return train, test


def evaluate(reasoner, tok, test, n_latent, limit, batch, use_calculator=True, show=2) -> dict:
    reasoner.eval()
    probs = test[: limit or len(test)]
    correct = calls = 0
    for lo in range(0, len(probs), batch):
        chunk = probs[lo: lo + batch]
        outs = reasoner.generate(tok, [p.question for p in chunk], n_latent,
                                 use_calculator=use_calculator)
        for j, (p, (text, c)) in enumerate(zip(chunk, outs)):
            ok = answer_of(text) == normalize_answer(p.answer)
            correct += ok
            calls += c
            if lo + j < show:
                print(f"  [eval n_latent={n_latent}] gold {p.answer} | {text.strip()[:200]!r}")
    n = len(probs)
    acc = correct / max(1, n)
    reasoner.train()
    return {"acc": acc, "stderr": (acc * (1 - acc) / max(1, n)) ** 0.5, "n": n,
            "n_latent": reasoner.slots(n_latent), "thought_mode": reasoner.thought_mode,
            "calculator_calls_per_problem": calls / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pretrained run dir, or hf://user/repo")
    ap.add_argument("--out", type=Path, required=True, help="latent run dir")
    ap.add_argument("--max_stage", type=int, default=4, help="last stage (steps replaced)")
    ap.add_argument("--c", type=int, default=2, help="latent thoughts per replaced step")
    ap.add_argument("--epochs_stage0", type=int, default=3)
    ap.add_argument("--epochs_per_stage", type=int, default=2)
    ap.add_argument("--thought_mode", default="latent", choices=THOUGHT_MODES,
                    help="latent = fed-back hidden state; pause/none = ablations")
    ap.add_argument("--data", default="gsm8k", choices=["gsm8k", "gsm8k_aug"],
                    help="gsm8k: GSM8K train (+ --metamath); gsm8k_aug: 385k equation-step "
                         "problems (Coconut's training set)")
    ap.add_argument("--aug_limit", type=int, default=0, help="use only this many GSM8K-Aug problems")
    ap.add_argument("--metamath", type=int, default=0, help="extra MetaMathQA GSM problems")
    ap.add_argument("--max_suffix_tokens", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--eval_limit", type=int, default=256)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every_minutes", type=float, default=15.0)
    ap.add_argument("--keep_ckpts", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile each block (fewer kernel launches in the thought loop)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    a.init = resolve_run(a.init)
    model, tok, base_step = load_run(a.init, a.device)
    ids = special_ids(tok)
    if not (out / "tokenizer.json").is_file():
        shutil.copyfile(a.init / "tokenizer.json", out / "tokenizer.json")
    reasoner = LatentReasoner(model, thought_mode=a.thought_mode,
                              pause_id=tok.token_to_id(PAUSE)).to(a.device).train()
    print(f"thought mode: {a.thought_mode}")

    train, test = load_problems(a)
    train = [p for p in train if len(encode(tok, p, 0, ids).suffix) <= a.max_suffix_tokens]

    # ---- resume -------------------------------------------------------------
    pos = {"stage": 0, "epoch": 0, "batch": 0}
    step, ck = 0, None
    if a.resume:
        ck, ck_path = load_latest_checkpoint(out, a.device)
    if ck is not None and "latent" in ck:
        if ck["latent"].get("thought_mode", "latent") != a.thought_mode:
            sys.exit(f"error: {out} was trained with --thought_mode "
                     f"{ck['latent'].get('thought_mode')}; use another --out")
        model.load_state_dict(ck["model"])
        reasoner.scale.data.fill_(ck["latent"]["scale"])
        pos, step = dict(ck["latent"]["pos"]), int(ck["step"])
        print(f"resumed {ck_path.name}: stage {pos['stage']} epoch {pos['epoch']} batch {pos['batch']}")
    else:
        s = reasoner.init_scale([encode(tok, p, 0, ids).prefix for p in train[:64]], a.chunk_size)
        print(f"init from {a.init} (pretrain step {base_step}); thought scale {s:.4f}")
        (out / "config.json").write_text(json.dumps(
            {"model": model.cfg.to_dict(), "latent_train": {k: str(v) for k, v in vars(a).items()},
             "init_step": base_step}, indent=2))

    dtype = resolve_dtype(a.dtype, a.device)
    dev_type = "cuda" if a.device.startswith("cuda") else "cpu"
    amp = (lambda: contextlib.nullcontext()) if dtype == torch.float32 else \
        (lambda: torch.autocast(device_type=dev_type, dtype=dtype))
    scaler = torch.amp.GradScaler(dev_type, enabled=dtype == torch.float16)
    print(f"precision: {str(dtype).replace('torch.', '')}, device {a.device}")
    if a.compile:
        for blk in model.blocks:
            blk.compile(dynamic=True)
        print("compiled blocks with torch.compile (first steps are slow while it compiles)")

    def new_optimizer():
        decay = [p for n, p in reasoner.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in reasoner.named_parameters() if p.ndim < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": a.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=a.lr, betas=(0.9, 0.95), fused=dev_type == "cuda")

    log_path = out / "log.jsonl"

    def log(rec: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def save(stage: int, stage_step: int, opt) -> None:
        save_checkpoint({
            "model": model.state_dict(), "config": model.cfg.to_dict(), "step": step,
            "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
            "latent": {"scale": float(reasoner.scale), "c": a.c, "pos": dict(pos),
                       "stage": stage, "n_latent": reasoner.slots(stage * a.c),
                       "thought_mode": a.thought_mode,
                       "stage_step": stage_step},
        }, out, keep_ckpts=a.keep_ckpts)

    B, G = a.batch_size, a.grad_accum
    last_save = time.time()
    for stage in range(pos["stage"], a.max_stage + 1):
        n_latent = stage * a.c
        epochs = a.epochs_stage0 if stage == 0 else a.epochs_per_stage
        opt = new_optimizer()
        stage_step = 0
        if (ck is not None and "latent" in ck and stage == pos["stage"]
                and int(ck["latent"].get("stage_step", 0)) > 0):  # mid-stage resume
            opt.load_state_dict(ck["optimizer"])
            if scaler.is_enabled() and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            stage_step = int(ck["latent"].get("stage_step", 0))
        print(f"\n===== stage {stage}: first {stage} steps -> {reasoner.slots(n_latent)} "
              f"{a.thought_mode} thoughts, {epochs} epoch(s)",
              flush=True)
        for epoch in range(pos["epoch"] if stage == pos["stage"] else 0, epochs):
            order = list(range(len(train)))
            random.Random(a.seed * 1000 + stage * 100 + epoch).shuffle(order)
            batches = [order[i: i + B] for i in range(0, len(order) - B + 1, B)]
            start = pos["batch"] if (stage == pos["stage"] and epoch == pos["epoch"]) else 0
            t0, tokens, run_loss = time.time(), 0, 0.0
            for bi in range(start, len(batches)):
                batch = [encode(tok, train[i], stage, ids) for i in batches[bi]]
                with amp():
                    loss = reasoner.loss(batch, n_latent, a.chunk_size)
                scaler.scale(loss / G).backward()
                run_loss += float(loss.detach())
                tokens += sum(len(b.suffix) for b in batch)
                if (bi + 1) % G:
                    continue
                lr = a.lr * min(1.0, (stage_step + 1) / a.warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(reasoner.parameters(), a.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                stage_step += 1
                pos.update(stage=stage, epoch=epoch, batch=bi + 1)
                if step % a.log_every == 0:
                    dt = time.time() - t0
                    l = run_loss / (a.log_every * G)
                    print(f"stage {stage} ep {epoch} {bi + 1}/{len(batches)} | step {step} | "
                          f"loss {l:.4f} | gnorm {float(gn):.2f} | scale {float(reasoner.scale):.3f} | "
                          f"{tokens / dt / 1e3:.1f}k tok/s", flush=True)
                    log({"step": step, "stage": stage, "epoch": epoch, "loss": l,
                         "grad_norm": float(gn), "scale": float(reasoner.scale), "lr": lr})
                    t0, tokens, run_loss = time.time(), 0, 0.0
                if time.time() - last_save > a.save_every_minutes * 60:
                    save(stage, stage_step, opt)
                    last_save = time.time()
                if a.max_steps and step >= a.max_steps:
                    save(stage, stage_step, opt)
                    print(f"stopping at --max_steps {a.max_steps}")
                    return
            pos.update(stage=stage, epoch=epoch + 1, batch=0)
        res = evaluate(reasoner, tok, test, n_latent, a.eval_limit, a.eval_batch)
        print(f"stage {stage} GSM8K ({res['n']}): acc {res['acc']:.3f} +/- {res['stderr']:.3f} "
              f"with {n_latent} thoughts", flush=True)
        log({"step": step, "stage": stage, "eval": res})
        pos.update(stage=stage + 1, epoch=0, batch=0)
        save(stage, 0, opt)
        last_save = time.time()
        ck = None
    print("done")


if __name__ == "__main__":
    main()
