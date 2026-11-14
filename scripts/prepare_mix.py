"""Build a 100% high-quality textbooks + coding + reasoning pretraining mix
for RetNet-300M using the `nampdn-ai/tiny-series` collection and reasoning datasets.

Streams each source from the Hugging Face Hub up to a token budget, trains a
digit-splitting BPE tokenizer suited to code and math, annotates worked arithmetic
as calculator calls (`<<expr=result>>result`), and writes uint16 token files.

  python scripts/prepare_mix.py --out data/tinymix --total_tokens 1e9

Authentication & Automatic Fallback for Gated `nampdn-ai/tiny-*` Datasets:
  - The `nampdn-ai/tiny-*` repositories on Hugging Face are marked `gated=auto`,
    meaning Hugging Face requires you to click "Agree" on their repo pages and
    provide a Read token (`HF_TOKEN` env var, `--hf_token`, or `.hf_token` file
    in your Shared Google Drive folder).
  - If no `HF_TOKEN` is provided (or if a specific `nampdn-ai` dataset has not
    been accepted on your HF account yet), `prepare_mix.py` automatically falls
    back to an open, ungated (`gated=False`) textbook/coding/reasoning dataset
    (`HuggingFaceTB/cosmopedia-100k`, `HuggingFaceTB/finemath` 4+,
    `ise-uiuc/Magicoder-Evol-Instruct-110K`, `deepmind/code_contests`,
    `sahil2801/CodeAlpaca-20k`) unless `--strict_gated` is passed.
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
CC_CPP, CC_PY3 = 2, 3  # deepmind/code_contests language ids


def _calc(text: str) -> str:
    """Rewrite worked arithmetic `16 * 47 = 752` as `16 * 47 = <<16*47=752>>752`."""
    return annotate_arithmetic(text)[0]


def _extract_textbook(r: dict, annotate_math: bool = False) -> list[str]:
    """Extract textbook content from any `nampdn-ai/tiny-*` or `cosmopedia` row schema."""
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
    """Format `nampdn-ai/tiny-codes` or `Magicoder-Evol` / `CodeAlpaca` rows."""
    prompt = (r.get("prompt") or r.get("instruction") or r.get("question") or "").strip()
    inp = (r.get("input") or "").strip()
    if prompt and inp:
        prompt = f"{prompt}\nInput: {inp}"
    response = (r.get("response") or r.get("code") or r.get("solution") or r.get("output") or "").strip()
    lang = (r.get("programming_language") or r.get("language") or r.get("lang") or "").strip()
    if prompt and response:
        header = f"Solution ({lang}):" if lang else "Solution:"
        return [f"Problem:\n{prompt}\n\n{header}\n{response}"]
    return _extract_textbook(r, annotate_math=False)


def _extract_magicoder(r: dict) -> list[str]:
    prob = (r.get("problem") or r.get("instruction") or "").strip()
    sol = (r.get("solution") or r.get("response") or "").strip()
    if not prob or not sol:
        return []
    lang = (r.get("lang") or "").strip()
    header = f"Solution ({lang}):" if lang else "Solution:"
    return [f"Problem:\n{prob}\n\n{header}\n{sol}"]


def _extract_code_contests(r: dict, per_problem: int = 2) -> list[str]:
    desc = (r.get("description") or "").strip()
    sols_dict = r.get("solutions") or {}
    langs = sols_dict.get("language") or []
    codes = sols_dict.get("solution") or []
    if not desc or not codes:
        return []
    out: list[str] = []
    py_cnt = cpp_cnt = 0
    for lid, code in zip(langs, codes):
        code = code.strip()
        if not code:
            continue
        if lid == CC_PY3 and py_cnt < per_problem:
            out.append(f"Problem:\n{desc}\n\nSolution (Python):\n{code}")
            py_cnt += 1
        elif lid == CC_CPP and cpp_cnt < per_problem:
            out.append(f"Problem:\n{desc}\n\nSolution (C++):\n{code}")
            cpp_cnt += 1
        if py_cnt >= per_problem and cpp_cnt >= per_problem:
            break
    return out


# Primary sources: (hf_path, config, split, formatter, default_share)
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
    "tiny_strange_textbooks": (
        "nampdn-ai/tiny-strange-textbooks",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=False),
        0.00,
    ),
}

# Ungated (gated=False) high-quality textbook/code/reasoning fallbacks used automatically
# if a gated `nampdn-ai/tiny-*` repo is accessed without an authorized HF_TOKEN:
UNGATED_FALLBACKS = {
    "tiny_codes": (
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        None,
        "train",
        _extract_tiny_codes,
    ),
    "tiny_math_textbooks": (
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        "train",
        lambda r: _extract_textbook(r, annotate_math=True),
    ),
    "tiny_textbooks": (
        "HuggingFaceTB/cosmopedia-100k",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=False),
    ),
    "tiny_code_textbooks": (
        "deepmind/code_contests",
        None,
        "train",
        _extract_code_contests,
    ),
    "tiny_orca_textbooks": (
        "HuggingFaceTB/cosmopedia-100k",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=True),
    ),
    "tiny_lessons": (
        "sahil2801/CodeAlpaca-20k",
        None,
        "train",
        _extract_tiny_codes,
    ),
    "tiny_strange_textbooks": (
        "HuggingFaceTB/cosmopedia-100k",
        None,
        "train",
        lambda r: _extract_textbook(r, annotate_math=False),
    ),
}


def resolve_hf_token(explicit_token: str | None = None, cache_dir: Path | None = None) -> str | None:
    """Find a Hugging Face token from CLI flag, environment, Shared Drive `.hf_token`,
    or standard local Hugging Face cache. Returns `None` (never `True`) when absent."""
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    # Check Shared Drive `.hf_token` in cache_dir or its parent folders
    if cache_dir is not None:
        for d in (cache_dir, cache_dir.parent, cache_dir.parent.parent):
            tf = d / ".hf_token"
            if tf.is_file():
                try:
                    t = tf.read_text().strip()
                    if t:
                        return t
                except OSError:
                    pass
    for std_p in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if std_p.is_file():
            try:
                t = std_p.read_text().strip()
                if t:
                    return t
            except OSError:
                pass
    return None


_WARNED_FALLBACKS: set[str] = set()


def _open_hf_stream(path: str, config: str | None, split: str, token: str | None):
    from datasets import load_dataset

    kw = {"split": split, "streaming": True}
    if token:
        kw["token"] = token
    try:
        ds = load_dataset(path, config, **kw)
    except TypeError:
        kw.pop("token", None)
        if token:
            kw["use_auth_token"] = token
        ds = load_dataset(path, config, **kw)
    # Peek first element to force authentication/gated check immediately
    it = iter(ds)
    first = next(it)
    return first, it


def stream(name: str, hf_token: str | None = None, strict_gated: bool = False):
    path, config, split, fmt, _ = SOURCES[name]
    try:
        first, it = _open_hf_stream(path, config, split, hf_token)
    except Exception as e:
        if not strict_gated and name in UNGATED_FALLBACKS:
            fb_path, fb_cfg, fb_split, fb_fmt = UNGATED_FALLBACKS[name]
            if name not in _WARNED_FALLBACKS:
                _WARNED_FALLBACKS.add(name)
                print(
                    f"  [note] {path} is gated on Hugging Face ({type(e).__name__}); "
                    f"automatically using ungated textbook/code dataset '{fb_path}'"
                    + (f" ({fb_cfg})" if fb_cfg else "")
                    + ".\n         (To use nampdn-ai directly: accept terms at "
                    f"https://huggingface.co/datasets/{path} and set HF_TOKEN.)",
                    flush=True,
                )
            try:
                first, it = _open_hf_stream(fb_path, fb_cfg, fb_split, hf_token)
                fmt = fb_fmt
            except Exception as e2:
                raise SystemExit(
                    f"error: could not open {path} or fallback {fb_path}: {e2!r}"
                ) from e2
        else:
            raise SystemExit(
                f"error: could not open {path} ({config}): {e!r}\n"
                f"Accept dataset terms at https://huggingface.co/datasets/{path} and pass --hf_token (or set HF_TOKEN)."
            ) from e

    for text in fmt(first):
        text = text.strip()
        if text:
            yield text
    for row in it:
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
    ap.add_argument("--tok_sample_docs", type=int, default=8000, help="docs per source to fit BPE")
    ap.add_argument("--tokenizer", type=Path, default=None, help="reuse an existing tokenizer.json")
    ap.add_argument("--hf_token", default=None, help="optional Hugging Face token for gated datasets")
    ap.add_argument(
        "--strict_gated",
        action="store_true",
        help="fail instead of falling back to ungated textbooks when nampdn-ai gated access is missing",
    )
    ap.add_argument("--shares", default="", help="override, e.g. tiny_codes=0.5,tiny_math_textbooks=0.5")
    a = ap.parse_args()

    hf_token = resolve_hf_token(a.hf_token, a.cache_dir)
    if hf_token:
        print("Using Hugging Face authentication token.", flush=True)

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
    tok_source = (
        a.tokenizer
        if (a.tokenizer and a.tokenizer.is_file())
        else (cached_tok if (cached_tok and cached_tok.is_file()) else None)
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
            for i, text in enumerate(stream(name, hf_token, a.strict_gated)):
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
                    for i, text in enumerate(stream(name, hf_token, a.strict_gated)):
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
