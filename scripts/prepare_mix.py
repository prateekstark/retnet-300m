"""Build a 100% high-quality textbooks + coding + reasoning pretraining mix
for RetNet-300M using the `nampdn-ai/tiny-series` collection and reasoning datasets.

Streams each source from the Hugging Face Hub up to a token budget, trains a
digit-splitting BPE tokenizer suited to code and math, annotates worked arithmetic
as calculator calls (`<<expr=result>>result`), and writes uint16 token files.

  python scripts/prepare_mix.py --out data/tinymix --total_tokens 1e9

Default Sources (100% Textbooks, Code & Step-by-Step Reasoning; NO raw web scrape):
  tiny_codes           nampdn-ai/tiny-codes                  Code + control-flow reasoning snippets   24%
  tiny_math_textbooks  nampdn-ai/tiny-math-textbooks         Synthetic math & logic textbooks         20%
  tiny_textbooks       nampdn-ai/tiny-textbooks              Foundational synthetic textbooks         14%
  tiny_code_textbooks  nampdn-ai/tiny-code-textbooks         Step-by-step code explanation textbooks  12%
  metamath             meta-math/MetaMathQA                  Math CoT Q&A (with <<a*b=c>> calc tags)  12%
  tiny_orca_textbooks  nampdn-ai/tiny-orca-textbooks         Step-by-step reasoning textbooks         10%
  magicoder            ise-uiuc/Magicoder-OSS-Instruct-75K   Algorithmic & coding problem solutions    4%
  tiny_lessons         nampdn-ai/tiny-lessons                Structured Markdown conceptual lessons    2%
  gsm8k                openai/gsm8k (main, TRAIN split only) Grade-school math with <<a*b=c>> calls    2%

Optional extra source (enable via --shares):
  tiny_strange_textbooks  nampdn-ai/tiny-strange-textbooks   Diverse niche synthetic textbooks         0%

Resumable across Colab disconnects / accounts:
  Pass `--cache_dir /content/drive/MyDrive/retnet-300m/data/tinymix_shards` to
  checkpoint each source shard as soon as it finishes tokenizing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.calculator import annotate_arithmetic  # noqa: E402

EOS = "<|endoftext|>"


def _calc(text: str) -> str:
    """Rewrite worked arithmetic `16 * 47 = 752` as `16 * 47 = <<16*47=752>>752`."""
    return annotate_arithmetic(text)[0]


def _extract_textbook(r: dict, annotate_math: bool = False) -> list[str]:
    """Extract textbook content from any `nampdn-ai/tiny-*` row schema.

    In `nampdn-ai/tiny-textbooks`, `r["textbook"]` holds the full synthesized
    textbook while `r["text"]` is the shorter seed snippet. In other subsets
    (`tiny-math-textbooks`, `tiny-code-textbooks`, `tiny-orca-textbooks`,
    `tiny-lessons`), the primary column is `text` or `textbook`.
    """
    tb = r.get("textbook")
    txt = r.get("text") or r.get("content") or r.get("lesson")
    chosen = None
    if isinstance(tb, str) and len(tb.strip()) > 30:
        if not isinstance(txt, str) or len(tb) >= len(txt):
            chosen = tb.strip()
    if chosen is None and isinstance(txt, str) and txt.strip():
        chosen = txt.strip()
    if not chosen:
        return []
    return [_calc(chosen) if annotate_math else chosen]


def _extract_tiny_codes(r: dict) -> list[str]:
    """Format `nampdn-ai/tiny-codes` rows (`prompt` + `response` + optional language)."""
    prompt = (r.get("prompt") or r.get("question") or r.get("instruction") or "").strip()
    response = (r.get("response") or r.get("code") or r.get("solution") or r.get("output") or "").strip()
    lang = (r.get("programming_language") or r.get("language") or r.get("lang") or "").strip()
    if prompt and response:
        header = f"Solution ({lang}):" if lang else "Solution:"
        return [f"Problem:\n{prompt}\n\n{header}\n{response}"]
    return _extract_textbook(r, annotate_math=False)


def _extract_magicoder(r: dict) -> list[str]:
    prob = (r.get("problem") or "").strip()
    sol = (r.get("solution") or "").strip()
    if not prob or not sol:
        return []
    lang = (r.get("lang") or "").strip()
    header = f"Solution ({lang}):" if lang else "Solution:"
    return [f"Problem:\n{prob}\n\n{header}\n{sol}"]


# name: (hf path, config, split, formatter -> list of texts, default share)
SOURCES = {
    "tiny_codes": (
        "nampdn-ai/tiny-codes",
        None,
        "train",
        _extract_tiny_codes,
        0.24,
    ),
    "tiny_math_textbooks": (
        "nampdn-ai/tiny-math-textbooks",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=True),
        0.20,
    ),
    "tiny_textbooks": (
        "nampdn-ai/tiny-textbooks",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=False),
        0.14,
    ),
    "tiny_code_textbooks": (
        "nampdn-ai/tiny-code-textbooks",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=False),
        0.12,
    ),
    "metamath": (
        "meta-math/MetaMathQA",
        None,
        "train",
        lambda r: [f"Question: {r['query'].strip()}\nAnswer: {_calc(r['response'].strip())}"],
        0.12,
    ),
    "tiny_orca_textbooks": (
        "nampdn-ai/tiny-orca-textbooks",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=True),
        0.10,
    ),
    "magicoder": (
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        None,
        "train",
        _extract_magicoder,
        0.04,
    ),
    "tiny_lessons": (
        "nampdn-ai/tiny-lessons",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=False),
        0.02,
    ),
    "gsm8k": (
        "openai/gsm8k",
        "main",
        "train",
        lambda r: [f"Question: {r['question'].strip()}\nAnswer: {r['answer'].strip()}"],
        0.02,
    ),
    # Optional extra source from nampdn-ai/tiny-series (0.0 by default; enable via --shares)
    "tiny_strange_textbooks": (
        "nampdn-ai/tiny-strange-textbooks",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=False),
        0.00,
    ),
}


def stream(name: str, hf_token: str | None = None):
    from datasets import load_dataset

    path, config, split, fmt, _ = SOURCES[name]
    token = hf_token or os.environ.get("HF_TOKEN") or True
    try:
        ds = load_dataset(path, config, split=split, streaming=True, token=token)
    except TypeError:
        ds = load_dataset(path, config, split=split, streaming=True)
    except Exception as e:
        raise SystemExit(
            f"error: could not open {path} ({config}): {e!r}\n"
            "If this dataset is gated on Hugging Face, accept its terms on huggingface.co "
            "and set HF_TOKEN (or pass --hf_token)."
        )
    for row in ds:
        for text in fmt(row):
            text = text.strip()
            if text:
                yield text


def make_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"<<|>>|="), behavior="isolated"),
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(sample_texts: list[str], vocab_size: int) -> Tokenizer:
    tok = make_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        min_frequency=2,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(sample_texts, trainer=trainer)
    return tok


class BinWriter:
    """Append token arrays to <name>.bin.tmp; rename to .bin only on success."""

    def __init__(self, path: Path):
        self.path, self.tmp = path, path.with_suffix(".bin.tmp")
        self.f = open(self.tmp, "wb")
        self.n = 0

    def write(self, arr: np.ndarray) -> None:
        arr.tofile(self.f)
        self.n += arr.size

    def write_file(self, src_bin: Path, expected_tokens: int) -> None:
        with open(src_bin, "rb") as src_f:
            shutil.copyfileobj(src_f, self.f, length=16 * 1024 * 1024)
        self.n += expected_tokens

    def commit(self) -> None:
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except OSError:
            pass
        self.f.close()
        self.tmp.replace(self.path)

    def abort(self) -> None:
        try:
            self.f.close()
            self.tmp.unlink(missing_ok=True)
        except OSError:
            pass


def encode(tok: Tokenizer, texts: list[str], eos_id: int) -> np.ndarray:
    ids: list[int] = []
    for e in tok.encode_batch(texts):
        ids.extend(e.ids)
        ids.append(eos_id)
    return np.asarray(ids, dtype=np.uint16)


def _valid_shard(cache_dir: Path, name: str, budget: int) -> dict | None:
    """Check if a completed per-source shard exists in `cache_dir`."""
    meta_p = cache_dir / f"{name}.meta.json"
    tr_p = cache_dir / f"{name}.train.bin"
    va_p = cache_dir / f"{name}.val.bin"
    if not (meta_p.is_file() and tr_p.is_file() and va_p.is_file()):
        return None
    try:
        info = json.loads(meta_p.read_text())
        n_tr = int(info["train_tokens"])
        n_va = int(info["val_tokens"])
        if tr_p.stat().st_size == n_tr * 2 and va_p.stat().st_size == n_va * 2:
            if n_tr >= int(0.95 * budget) or int(info.get("passes", 0)) > 50:
                return info
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/tinymix"))
    ap.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="optional directory (e.g. on Shared Drive) to cache completed per-source shards across disconnects",
    )
    ap.add_argument("--total_tokens", type=float, default=1e9)
    ap.add_argument("--val_tokens_per_source", type=float, default=1e6)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--tok_sample_docs", type=int, default=15000, help="docs per source to fit BPE")
    ap.add_argument("--tokenizer", type=Path, default=None, help="reuse an existing tokenizer.json")
    ap.add_argument("--hf_token", default=None, help="optional Hugging Face token for gated datasets")
    ap.add_argument("--shares", default="", help="override, e.g. tiny_codes=0.5,tiny_math_textbooks=0.5")
    a = ap.parse_args()

    shares = {k: v[4] for k, v in SOURCES.items() if v[4] > 0}
    if a.shares:
        shares = {
            k.strip(): float(v)
            for k, v in (kv.split("=") for kv in a.shares.split(",") if kv.strip())
            if float(v) > 0
        }
        unknown = set(shares) - set(SOURCES)
        if unknown:
            raise SystemExit(f"error: unknown sources {unknown}; choose from {list(SOURCES)}")

    try:
        a.out.mkdir(parents=True, exist_ok=True)
        if a.cache_dir is not None:
            a.cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"error: cannot create output/cache directory: {e}")

    # ---- tokenizer (reuse from --tokenizer, or --cache_dir/tokenizer.json, or train fresh)
    cached_tok = a.cache_dir / "tokenizer.json" if a.cache_dir else None
    tok_source = a.tokenizer if (a.tokenizer and a.tokenizer.is_file()) else (
        cached_tok if (cached_tok and cached_tok.is_file()) else None
    )
    if tok_source:
        try:
            tok = Tokenizer.from_file(str(tok_source))
            print(f"reusing tokenizer from {tok_source} (vocab {tok.get_vocab_size()})", flush=True)
        except Exception as e:
            raise SystemExit(f"error: could not load tokenizer {tok_source}: {e!r}")
    else:
        print("collecting tokenizer sample from tiny-series & reasoning sources ...", flush=True)
        sample = []
        for name in shares:
            for i, text in enumerate(stream(name, a.hf_token)):
                if i >= a.tok_sample_docs:
                    break
                sample.append(text[:20000])
        print(f"training {a.vocab_size}-token digit-splitting BPE on {len(sample)} docs ...", flush=True)
        tok = train_tokenizer(sample, a.vocab_size)
        del sample

    if tok.get_vocab_size() > 65535 or tok.token_to_id(EOS) is None:
        raise SystemExit("error: tokenizer must fit uint16 and contain <|endoftext|>")
    eos_id = tok.token_to_id(EOS)
    try:
        tok.save(str(a.out / "tokenizer.json"))
        if cached_tok is not None and not cached_tok.is_file():
            tok.save(str(cached_tok))
    except Exception as e:
        raise SystemExit(f"error: could not save tokenizer: {e!r}")
    demo = "Natalia sold 48/2 = <<48/2=24>>24 clips."
    print("example:", tok.encode(demo).tokens, flush=True)

    # ---- token files
    train, val = BinWriter(a.out / "train.bin"), BinWriter(a.out / "val.bin")
    counts: dict[str, dict[str, int | list[int]]] = {}
    try:
        for name, share in shares.items():
            budget = int(share * a.total_tokens)
            val_budget = int(a.val_tokens_per_source)
            val_start = val.n

            # Check per-source shard cache if enabled
            if a.cache_dir is not None:
                cached_info = _valid_shard(a.cache_dir, name, budget)
                if cached_info is not None:
                    n_tr = int(cached_info["train_tokens"])
                    n_va = int(cached_info["val_tokens"])
                    epochs = int(cached_info["passes"])
                    train.write_file(a.cache_dir / f"{name}.train.bin", n_tr)
                    val.write_file(a.cache_dir / f"{name}.val.bin", n_va)
                    print(
                        f"  {name}: restored from shard cache ({n_tr / 1e6:.1f}M train, "
                        f"{n_va / 1e6:.2f}M val, {epochs} pass(es))",
                        flush=True,
                    )
                    counts[name] = {
                        "train_tokens": n_tr,
                        "val_tokens": n_va,
                        "passes": epochs,
                        "val_range": [val_start, val.n],
                    }
                    continue

            shard_tr = BinWriter(a.cache_dir / f"{name}.train.bin") if a.cache_dir else None
            shard_va = BinWriter(a.cache_dir / f"{name}.val.bin") if a.cache_dir else None
            n_train = n_val = 0
            epochs = 0
            t0 = time.time()
            val_docs = 0
            try:
                while n_train < budget:
                    epochs += 1
                    buf = []
                    for i, text in enumerate(stream(name, a.hf_token)):
                        if epochs > 1 and i < val_docs:
                            continue
                        buf.append(text)
                        if len(buf) < 512:
                            continue
                        arr = encode(tok, buf, eos_id)
                        to_val = epochs == 1 and n_val < val_budget
                        val_docs += len(buf) if to_val else 0
                        buf = []
                        if to_val:
                            val.write(arr)
                            if shard_va:
                                shard_va.write(arr)
                            n_val += arr.size
                        else:
                            train.write(arr)
                            if shard_tr:
                                shard_tr.write(arr)
                            n_train += arr.size
                            if n_train >= budget:
                                break
                        rate = (n_train + n_val) / max(1e-9, time.time() - t0)
                        print(
                            f"\r  {name}: {n_train / 1e6:8.1f}M / {budget / 1e6:.0f}M train tokens"
                            f"  ({rate / 1e3:.0f}k tok/s)",
                            end="",
                            flush=True,
                        )
                    if buf and n_train < budget:
                        arr = encode(tok, buf, eos_id)
                        train.write(arr)
                        if shard_tr:
                            shard_tr.write(arr)
                        n_train += arr.size
                    if epochs > 50:
                        break  # tiny source; stop repeating
                if shard_tr and shard_va and a.cache_dir:
                    shard_tr.commit()
                    shard_va.commit()
                    (a.cache_dir / f"{name}.meta.json").write_text(
                        json.dumps({"train_tokens": n_train, "val_tokens": n_val, "passes": epochs})
                    )
            except BaseException:
                if shard_tr:
                    shard_tr.abort()
                if shard_va:
                    shard_va.abort()
                raise

            print(
                f"\n  {name}: {n_train / 1e6:.1f}M train, {n_val / 1e6:.2f}M val, {epochs} pass(es)",
                flush=True,
            )
            counts[name] = {
                "train_tokens": n_train,
                "val_tokens": n_val,
                "passes": epochs,
                "val_range": [val_start, val.n],
            }
        train.commit()
        val.commit()
    except BaseException:
        train.abort()
        val.abort()
        raise

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "eos_id": eos_id,
        "dtype": "uint16",
        "train_tokens": train.n,
        "val_tokens": val.n,
        "sources": counts,
        "shares": shares,
    }
    try:
        (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    except OSError as e:
        raise SystemExit(f"error: could not write meta.json: {e}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        sys.exit(f"error: {e}\n(check the output path and free disk space)")
