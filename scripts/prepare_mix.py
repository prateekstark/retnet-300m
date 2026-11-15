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
  - A source whose rows are stubs (e.g. `nampdn-ai/tiny-math-textbooks`, whose rows
    are still "[Not ready yet!]" placeholders) is also replaced by its fallback, and a
    shard averaging < 32 tokens/doc is rejected. The choice is saved in
    `<cache_dir>/<source>.source.json` so every session uses the same dataset.

Resuming across Colab disconnects and accounts:
  - Each source is tokenized into its own shard in `--cache_dir` (on Shared Drive).
  - Every 30 s the partial shard is fsynced and `<source>.progress.json` records the
    rows consumed and tokens written; a new session truncates the partial files to
    that point and continues streaming from the same row.
  - Finished shards are reused; `train.bin`/`val.bin` are concatenated at the end.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# Silence noisy huggingface_hub warning ("Repo card metadata block was not found. Setting CardData to empty.")
logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.calculator import annotate_arithmetic  # noqa: E402

EOS = "<|endoftext|>"
CC_CPP, CC_PY3 = 2, 3  # deepmind/code_contests language ids
_META_KEYS = frozenset(
    {
        "id",
        "url",
        "source",
        "domain",
        "topic",
        "subtopic",
        "subject",
        "title",
        "programming_language",
        "language",
        "lang",
        "model",
        "audience",
    }
)


def _calc(text: str) -> str:
    """Rewrite worked arithmetic `16 * 47 = 752` as `16 * 47 = <<16*47=752>>752`."""
    return annotate_arithmetic(text)[0]


def _extract_textbook(r: dict, annotate_math: bool = False) -> list[str]:
    """Extract textbook content from any `nampdn-ai/tiny-*` or `cosmopedia`/`finemath` row schema.

    Picks the longest substantive string across known textbook columns or any non-metadata string
    column, and filters out short/stub strings (< 60 chars) such as bare titles or topic tags.
    """
    candidates: list[str] = []
    for k in ("textbook", "markdown", "text", "content", "lesson", "body", "document", "output", "response"):
        v = r.get(k)
        if isinstance(v, str) and len(v.strip()) >= 60:
            candidates.append(v.strip())
    if not candidates:
        for k, v in r.items():
            if k not in _META_KEYS and isinstance(v, str) and len(v.strip()) >= 60:
                candidates.append(v.strip())
    if not candidates:
        return []
    chosen = max(candidates, key=len)
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


SHARD_VERSION = 2
MAX_PASSES = 50  # tiny sources (e.g. gsm8k) are repeated at most this many times
MIN_TOKENS_PER_DOC = 32  # below this a source is emitting stub rows, not documents
_PROBE_ROWS = 16


@dataclass(frozen=True)
class Source:
    """One concrete Hugging Face dataset backing a mix entry."""

    kind: str  # "primary" or "fallback"
    path: str
    config: str | None
    split: str
    fmt: Callable[[dict], list[str]]


def get_source(name: str, kind: str) -> Source:
    if kind == "primary":
        path, config, split, fmt, _ = SOURCES[name]
    elif kind == "fallback":
        path, config, split, fmt = UNGATED_FALLBACKS[name]
    else:
        raise ValueError(f"unknown source kind {kind!r}")
    return Source(kind, path, config, split, fmt)


def _load_stream(src: Source, token: str | None):
    from datasets import load_dataset

    kw = {"split": src.split, "streaming": True}
    if token:
        kw["token"] = token
    try:
        return load_dataset(src.path, src.config, **kw)
    except TypeError:  # very old `datasets`
        kw.pop("token", None)
        if token:
            kw["use_auth_token"] = token
        return load_dataset(src.path, src.config, **kw)


def iter_rows(src: Source, token: str | None, skip: int = 0) -> Iterator[dict]:
    """Stream raw rows of `src`, starting after the first `skip` rows."""
    ds = _load_stream(src, token)
    if skip:
        try:
            ds = ds.skip(skip)
        except AttributeError:
            return itertools.islice(iter(ds), skip, None)
    return iter(ds)


def _texts(fmt: Callable[[dict], list[str]], row: dict) -> list[str]:
    return [t for t in (x.strip() for x in fmt(row)) if t]


def _probe(src: Source, token: str | None, min_mean_chars: float = 200.0) -> str | None:
    """None if `src` opens and yields real documents, else the reason it does not."""
    try:
        rows = list(itertools.islice(iter_rows(src, token), _PROBE_ROWS))
    except Exception as e:  # gated, missing, network
        return f"cannot open: {type(e).__name__}: {str(e)[:160]}"
    texts = [t for r in rows for t in _texts(src.fmt, r)]
    mean = sum(map(len, texts)) / len(texts) if texts else 0.0
    if not rows or len(texts) < max(1, len(rows) // 2) or mean < min_mean_chars:
        return f"stub rows (avg {mean:.0f} chars over {len(rows)} rows)"
    return None


_RESOLVED: dict[str, Source] = {}


def resolve_source(
    name: str, token: str | None, strict_gated: bool, cache_dir: Path | None
) -> Source:
    """Pick the primary dataset for `name`, or its ungated fallback when the
    primary is gated/unavailable or emits stub rows. The choice is remembered in
    `cache_dir/<name>.source.json` so every session (and account) keeps
    building a shard from the same dataset."""
    if name in _RESOLVED:
        return _RESOLVED[name]
    choice = cache_dir / f"{name}.source.json" if cache_dir else None
    if choice is not None and choice.is_file():
        try:
            kind = json.loads(choice.read_text())["kind"]
            src = get_source(name, kind)
            _RESOLVED[name] = src
            return src
        except Exception:
            pass
    src = get_source(name, "primary")
    # Textbook sources must yield long documents; short Q&A sources only non-empty ones.
    reason = _probe(src, token, 200.0 if name in UNGATED_FALLBACKS else 1.0)
    if reason is not None:
        if strict_gated or name not in UNGATED_FALLBACKS:
            raise SystemExit(
                f"error: {src.path}: {reason}\n"
                f"Accept the terms at https://huggingface.co/datasets/{src.path} and set HF_TOKEN, "
                "or drop this source with --shares."
            )
        fb = get_source(name, "fallback")
        print(f"  [note] {src.path}: {reason}; using ungated '{fb.path}'"
              + (f" ({fb.config})" if fb.config else ""), flush=True)
        fb_reason = _probe(fb, token)
        if fb_reason is not None:
            raise SystemExit(f"error: fallback {fb.path} also unusable: {fb_reason}")
        src = fb
    if choice is not None:
        _write_json(choice, {"kind": src.kind, "path": src.path, "config": src.config})
    _RESOLVED[name] = src
    return src


def stream(name: str, hf_token: str | None = None, strict_gated: bool = False,
           cache_dir: Path | None = None) -> Iterator[str]:
    """Documents of mix entry `name` (used for the tokenizer sample)."""
    src = resolve_source(name, hf_token, strict_gated, cache_dir)
    for row in iter_rows(src, hf_token):
        yield from _texts(src.fmt, row)


def _prefetch(it: Iterator, depth: int = 8192) -> Iterator:
    """Pull rows on a background thread so downloading overlaps tokenizing."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    err: list[BaseException] = []
    stop = threading.Event()

    def work():
        try:
            for x in it:
                if stop.is_set():
                    return
                q.put(x)
        except BaseException as e:  # surfaced in the consumer
            err.append(e)
        finally:
            q.put(done)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        while True:
            x = q.get()
            if x is done:
                if err:
                    raise err[0]
                return
            yield x
    finally:
        stop.set()
        for _ in range(100):  # unblock a producer stuck on a full queue
            if not th.is_alive():
                break
            try:
                q.get(timeout=0.05)
            except queue.Empty:
                pass


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


def _write_json(path: Path, obj: dict) -> None:
    """Write JSON via tmp+replace; fall back to in-place on Drive files owned
    by another Google account (where replace is denied)."""
    text = json.dumps(obj, indent=2)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        path.write_text(text)


def _fsync(f) -> None:
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass


def _remove(path: Path) -> None:
    """Truncate then unlink (frees Drive quota even if unlink only trashes it)."""
    try:
        if path.is_file():
            with open(path, "r+b") as f:
                f.truncate(0)
        path.unlink(missing_ok=True)
    except OSError:
        pass


def encode(tok: Tokenizer, texts: list[str], eos_id: int) -> np.ndarray:
    ids: list[int] = []
    for e in tok.encode_batch(texts):
        ids.extend(e.ids)
        ids.append(eos_id)
    return np.asarray(ids, dtype=np.uint16)


def _open_part(path: Path, n_tokens: int) -> tuple[Path, object]:
    """Open `path` for appending after exactly `n_tokens` uint16 tokens.

    If this account may not write a file another account created, continue in
    a copy this account owns."""
    try:
        f = open(path, "r+b") if path.exists() else open(path, "w+b")
    except OSError:
        new = path.with_name(f"{path.stem}.{os.getpid()}{path.suffix}")
        shutil.copyfile(path, new)
        path, f = new, open(new, "r+b")
    f.truncate(n_tokens * 2)
    f.seek(n_tokens * 2)
    return path, f


def _shard_paths(shard_dir: Path, name: str) -> dict[str, Path]:
    return {
        "meta": shard_dir / f"{name}.meta.json",
        "train": shard_dir / f"{name}.train.bin",
        "val": shard_dir / f"{name}.val.bin",
        "progress": shard_dir / f"{name}.progress.json",
    }


def _count_docs(path: Path, eos_id: int, chunk: int = 1 << 26) -> int:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    return sum(int(np.count_nonzero(data[i : i + chunk] == eos_id)) for i in range(0, len(data), chunk))


def valid_shard(shard_dir: Path, name: str, budget: int, eos_id: int) -> dict | None:
    """The meta of a finished, size-verified, non-stub shard for `name`, else None."""
    p = _shard_paths(shard_dir, name)
    if not (p["meta"].is_file() and p["train"].is_file() and p["val"].is_file()):
        return None
    try:
        info = json.loads(p["meta"].read_text())
        n_tr, n_va = int(info["train_tokens"]), int(info["val_tokens"])
        if p["train"].stat().st_size != n_tr * 2 or p["val"].stat().st_size != n_va * 2:
            return None
        if n_tr < int(0.95 * budget) and int(info.get("passes", 0)) < MAX_PASSES:
            return None
        docs = info.get("docs")
        if docs is None:  # shard from an older prepare_mix: verify it is not stub rows
            docs = _count_docs(p["train"], eos_id)
            info["docs"] = docs
        if n_tr / max(1, docs) < MIN_TOKENS_PER_DOC:
            print(f"  {name}: cached shard has {n_tr / max(1, docs):.0f} tokens/doc (stub rows); rebuilding",
                  flush=True)
            return None
        return info
    except Exception:
        return None


def build_shard(
    name: str,
    src: Source,
    tok: Tokenizer,
    eos_id: int,
    budget: int,
    val_budget: int,
    shard_dir: Path,
    hf_token: str | None = None,
    batch_docs: int = 1024,
    save_every_s: float = 30.0,
    log_every_s: float = 30.0,
) -> dict:
    """Tokenize source `name` into `shard_dir/<name>.{train,val}.bin`.

    Progress (rows consumed, tokens written) is saved every `save_every_s`
    seconds next to the partial files, so a Colab disconnect costs at most
    that much work: the next session truncates the partial files back to the
    last saved progress and continues streaming from the same row.
    """
    p = _shard_paths(shard_dir, name)
    fresh = {"version": SHARD_VERSION, "source": src.kind, "path": src.path, "pass": 1, "rows": 0,
             "val_rows": 0, "n_train": 0, "n_val": 0, "docs": 0,
             "train_part": f"{name}.train.part", "val_part": f"{name}.val.part"}
    st = dict(fresh)
    if p["progress"].is_file():
        try:
            prev = json.loads(p["progress"].read_text())
            tr, va = shard_dir / prev["train_part"], shard_dir / prev["val_part"]
            if (prev.get("version") == SHARD_VERSION and prev.get("path") == src.path
                    and tr.is_file() and va.is_file()
                    and tr.stat().st_size >= 2 * prev["n_train"] and va.stat().st_size >= 2 * prev["n_val"]):
                st = prev
                print(f"  {name}: resuming at pass {st['pass']}, row {st['rows']:,} "
                      f"({st['n_train'] / 1e6:.1f}M train tokens already done)", flush=True)
        except Exception:
            st = dict(fresh)

    tr_path, ftr = _open_part(shard_dir / st["train_part"], st["n_train"])
    va_path, fva = _open_part(shard_dir / st["val_part"], st["n_val"])
    st["train_part"], st["val_part"] = tr_path.name, va_path.name

    def save_progress():
        _fsync(ftr)
        _fsync(fva)
        _write_json(p["progress"], st)

    t0 = last_save = last_log = time.time()
    tok0 = st["n_train"] + st["n_val"]
    try:
        while st["n_train"] < budget and st["pass"] <= MAX_PASSES:
            rows_this_pass = 0
            buf: list[str] = []
            buf_rows = 0

            def flush() -> None:
                nonlocal buf, buf_rows
                arr = encode(tok, buf, eos_id)
                st["rows"] += buf_rows
                st["docs"] += len(buf)
                if st["pass"] == 1 and st["n_val"] < val_budget:
                    arr.tofile(fva)
                    st["n_val"] += arr.size
                    st["val_rows"] = st["rows"]
                else:
                    arr.tofile(ftr)
                    st["n_train"] += arr.size
                buf, buf_rows = [], 0

            for row in _prefetch(iter_rows(src, hf_token, skip=st["rows"])):
                rows_this_pass += 1
                buf_rows += 1
                buf.extend(_texts(src.fmt, row))
                # small batches while filling val, so tiny datasets keep most rows for train
                filling_val = st["pass"] == 1 and st["n_val"] < val_budget
                if len(buf) < (min(64, batch_docs) if filling_val else batch_docs):
                    continue
                flush()
                if st["n_train"] >= budget:
                    break
                now = time.time()
                if now - last_save >= save_every_s:
                    save_progress()
                    last_save = now
                if now - last_log >= log_every_s:
                    rate = (st["n_train"] + st["n_val"] - tok0) / max(1e-9, now - t0)
                    eta = (budget - st["n_train"]) / max(rate, 1.0)
                    print(f"  {name}: {st['n_train'] / 1e6:7.1f}M / {budget / 1e6:.0f}M train tokens "
                          f"| {rate / 1e3:.0f}k tok/s | pass {st['pass']} | ETA {eta / 60:.0f} min", flush=True)
                    last_log = now
                if st["docs"] >= 20_000 and (st["n_train"] + st["n_val"]) / st["docs"] < MIN_TOKENS_PER_DOC:
                    raise SystemExit(
                        f"error: {src.path} averages {(st['n_train'] + st['n_val']) / st['docs']:.0f} tokens/doc "
                        "(stub rows). Delete its .source.json/.progress.json in the shard cache and pass "
                        f"--shares without {name}, or rerun without --strict_gated to use the fallback."
                    )
            else:  # end of one pass over the dataset
                if buf_rows:
                    flush()
                if rows_this_pass == 0:
                    raise SystemExit(f"error: {src.path} has no rows left to train on (all held out for val?)")
                if st["n_train"] < budget:
                    st["pass"] += 1
                    st["rows"] = st["val_rows"]  # later passes skip the rows held out for val
                save_progress()
                last_save = time.time()
                continue
            save_progress()
        _fsync(ftr)
        _fsync(fva)
    finally:
        ftr.close()
        fva.close()

    # Finalize: parts become the shard; meta.json is written last (it marks completion).
    for part, final in ((tr_path, p["train"]), (va_path, p["val"])):
        try:
            os.replace(part, final)
        except OSError:  # another account owns `final`: overwrite in place
            shutil.copyfile(part, final)
            _remove(part)
    info = {"version": SHARD_VERSION, "source": src.path, "train_tokens": st["n_train"],
            "val_tokens": st["n_val"], "passes": st["pass"], "docs": st["docs"]}
    _write_json(p["meta"], info)
    _remove(p["progress"])
    print(f"  {name}: done, {st['n_train'] / 1e6:.1f}M train, {st['n_val'] / 1e6:.2f}M val, "
          f"{st['pass']} pass(es), {(st['n_train'] + st['n_val']) / max(1, st['docs']):.0f} tokens/doc",
          flush=True)
    return info


def _cleanup_legacy(shard_dir: Path) -> None:
    """Remove partial shards left by older prepare_mix versions (*.bin.tmp)."""
    for f in shard_dir.glob("*.bin.tmp"):
        print(f"  removing stale partial shard {f.name}", flush=True)
        _remove(f)


def _concat(out: Path, parts: list[tuple[Path, int]]) -> int:
    """Concatenate token files into `out` (written as .tmp, renamed on success)."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    n = 0
    with open(tmp, "wb") as f:
        for src, n_tok in parts:
            with open(src, "rb") as g:
                shutil.copyfileobj(g, f, length=16 * 1024 * 1024)
            n += n_tok
        _fsync(f)
    if tmp.stat().st_size != 2 * n:
        raise SystemExit(f"error: {out} size mismatch after concatenation")
    tmp.replace(out)
    return n


def parse_shares(spec: str) -> dict[str, float]:
    if not spec:
        return {k: v[4] for k, v in SOURCES.items() if v[4] > 0}
    shares = {}
    for kv in spec.split(","):
        if not kv.strip():
            continue
        k, _, v = kv.partition("=")
        if float(v) > 0:
            shares[k.strip()] = float(v)
    unknown = set(shares) - set(SOURCES)
    if unknown:
        raise SystemExit(f"error: unknown sources {sorted(unknown)}; choose from {list(SOURCES)}")
    return shares


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/tinymix"))
    ap.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="directory (e.g. on Shared Drive) holding per-source shards and their resume progress; "
        "default <out>/shards",
    )
    ap.add_argument("--total_tokens", type=float, default=1e9)
    ap.add_argument("--val_tokens_per_source", type=float, default=1e6)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--tok_sample_docs", type=int, default=8000, help="docs per source to fit BPE")
    ap.add_argument("--tokenizer", type=Path, default=None, help="reuse an existing tokenizer.json")
    ap.add_argument("--hf_token", default=None, help="Hugging Face token (prefer the HF_TOKEN env var)")
    ap.add_argument(
        "--strict_gated",
        action="store_true",
        help="fail instead of falling back to ungated datasets when a nampdn-ai source is gated or stub",
    )
    ap.add_argument("--shares", default="", help="override, e.g. tiny_codes=0.5,tiny_math_textbooks=0.5")
    a = ap.parse_args()

    shares = parse_shares(a.shares)
    shard_dir = a.cache_dir or (a.out / "shards")
    try:
        a.out.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"error: cannot create output/cache directory: {e}")
    _cleanup_legacy(shard_dir)

    hf_token = resolve_hf_token(a.hf_token, shard_dir)
    print(f"Hugging Face token: {'found' if hf_token else 'none (gated sources use ungated fallbacks)'}",
          flush=True)

    # ---- tokenizer: --tokenizer, else the shard cache's, else train one
    cached_tok = shard_dir / "tokenizer.json"
    tok_source = a.tokenizer if (a.tokenizer and a.tokenizer.is_file()) else (
        cached_tok if cached_tok.is_file() else None)
    if tok_source:
        try:
            tok = Tokenizer.from_file(str(tok_source))
        except Exception as e:
            raise SystemExit(f"error: could not load tokenizer {tok_source}: {e!r}")
        print(f"reusing tokenizer {tok_source} (vocab {tok.get_vocab_size()})", flush=True)
    else:
        print("collecting tokenizer sample ...", flush=True)
        sample = []
        for name in shares:
            src = resolve_source(name, hf_token, a.strict_gated, shard_dir)
            n0 = len(sample)
            for row in iter_rows(src, hf_token):
                sample.extend(t[:20000] for t in _texts(src.fmt, row))
                if len(sample) - n0 >= a.tok_sample_docs:
                    break
            print(f"  {name}: {len(sample) - n0} docs from {src.path}", flush=True)
        print(f"training {a.vocab_size}-token digit-splitting BPE on {len(sample)} docs ...", flush=True)
        tok = train_tokenizer(sample, a.vocab_size)
        del sample
    if tok.get_vocab_size() > 65535 or tok.token_to_id(EOS) is None:
        raise SystemExit("error: tokenizer must fit uint16 and contain <|endoftext|>")
    eos_id = tok.token_to_id(EOS)
    try:
        tok.save(str(a.out / "tokenizer.json"))
        if not cached_tok.is_file():
            tok.save(str(cached_tok))
    except Exception as e:
        raise SystemExit(f"error: could not save tokenizer: {e!r}")
    print("example:", tok.encode("Natalia sold 48/2 = <<48/2=24>>24 clips.").tokens, flush=True)

    # ---- one resumable shard per source
    counts: dict[str, dict] = {}
    for name, share in shares.items():
        budget = int(share * a.total_tokens)
        info = valid_shard(shard_dir, name, budget, eos_id)
        if info is not None:
            print(f"  {name}: cached shard ({info['train_tokens'] / 1e6:.1f}M train, "
                  f"{info['val_tokens'] / 1e6:.2f}M val)", flush=True)
        else:
            src = resolve_source(name, hf_token, a.strict_gated, shard_dir)
            print(f"  {name}: building from {src.path} (budget {budget / 1e6:.0f}M tokens)", flush=True)
            info = build_shard(name, src, tok, eos_id, budget, int(a.val_tokens_per_source),
                               shard_dir, hf_token)
        counts[name] = info

    # ---- concatenate shards into train.bin / val.bin (meta.json written last)
    print("concatenating shards ...", flush=True)
    paths = {n: _shard_paths(shard_dir, n) for n in counts}
    n_train = _concat(a.out / "train.bin", [(paths[n]["train"], c["train_tokens"]) for n, c in counts.items()])
    _concat(a.out / "val.bin", [(paths[n]["val"], c["val_tokens"]) for n, c in counts.items()])
    pos, sources = 0, {}
    for n, c in counts.items():
        sources[n] = {"train_tokens": c["train_tokens"], "val_tokens": c["val_tokens"],
                      "passes": c["passes"], "docs": c.get("docs"), "source": c.get("source"),
                      "val_range": [pos, pos + c["val_tokens"]]}
        pos += c["val_tokens"]
    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": eos_id, "dtype": "uint16",
            "train_tokens": n_train, "val_tokens": pos, "sources": sources, "shares": shares}
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
