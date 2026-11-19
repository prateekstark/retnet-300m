"""Calculator-in-the-loop generation (the GSM8K "<<expr=result>>" convention).

Training data writes arithmetic as  <<48/2=24>>24.  At inference, when the model
has written  <<48/2=  we compute 24 exactly, insert "24>>", and let it continue.
With RetNet this is cheap: inserted tokens just extend the recurrent state; no
KV cache is rebuilt.

The evaluator is a small arithmetic-only parser over Python's AST. It never
calls eval(): only numbers, + - * / // % **, unary minus and parentheses are
accepted, with limits on exponent size and result magnitude.
"""

from __future__ import annotations

import ast
import operator
import re

import torch

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_MAX_ABS = 1e15
_OPEN = re.compile(r"<<([^<>]*)=$")  # an open calculator call ending in '='


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 10 or abs(left) > 1e6):
            raise ValueError("exponent too large")
        out = _BINOPS[type(node.op)](left, right)
        if abs(out) > _MAX_ABS:
            raise ValueError("result too large")
        return out
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


def safe_eval(expr: str) -> str | None:
    """'48/2' -> '24'; '7/3' -> '2.3333'; anything else (names, calls, ...) -> None."""
    expr = expr.replace(",", "").replace("x", "*").replace("×", "*").replace("÷", "/").strip()
    if not expr or len(expr) > 200:
        return None
    try:
        val = _eval(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError):
        return None
    if isinstance(val, float):
        if val != val or val in (float("inf"), float("-inf")):
            return None
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val)


@torch.no_grad()
def generate_with_calculator(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = 0,
    stop: tuple[str, ...] = (),
    use_calculator: bool = True,
) -> tuple[str, int]:
    """Decode one prompt in recurrent mode, filling in <<expr= with exact results.

    Returns (generated_text, calculator_calls). temperature=0 is greedy.
    """
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt).ids
    logits, _, state = model(torch.tensor([ids], device=device), mode="chunkwise", chunk_size=256)
    pos = len(ids)
    out: list[int] = []
    calls = 0
    next_logits = logits[0, -1]

    def feed(tokens: list[int]):
        nonlocal state, pos, next_logits
        lg, _, state = model(
            torch.tensor([tokens], device=device), state=state, offset=pos, mode="recurrent"
        )
        pos += len(tokens)
        next_logits = lg[0, -1]

    while len(out) < max_new_tokens:
        if temperature <= 0:
            tok = int(next_logits.argmax())
        else:
            lg = next_logits / temperature
            if top_k:
                lg = lg.masked_fill(
                    lg < torch.topk(lg, min(top_k, lg.numel())).values[-1], float("-inf")
                )
            tok = int(torch.multinomial(torch.softmax(lg, -1), 1))
        if eos_id is not None and tok == eos_id:
            break
        out.append(tok)
        feed([tok])
        text = tokenizer.decode(out)
        if any(text.endswith(s) for s in stop):
            break
        inject = _calculator_tokens(text, tokenizer) if use_calculator else None
        if inject:
            out.extend(inject)
            feed(inject)
            calls += 1
    return tokenizer.decode(out), calls


def _calculator_tokens(text: str, tokenizer) -> list[int] | None:
    """If `text` ends in an open call "<<expr=", the tokens of "result>>"."""
    m = _OPEN.search(text)
    if not m:
        return None
    result = safe_eval(m.group(1))
    return tokenizer.encode(result + ">>").ids if result is not None else None


def _cat_states(states: list) -> list:
    """Stack per-sequence model states (lists of per-layer S or (S, conv_cache))."""
    out = []
    for layer in zip(*states):
        if isinstance(layer[0], tuple):
            out.append(tuple(torch.cat(parts, dim=0) for parts in zip(*layer)))
        else:
            out.append(torch.cat(layer, dim=0))
    return out


@torch.no_grad()
def generate_batch_greedy(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    eos_id: int | None = 0,
    stop: tuple[str, ...] = (),
    stop_regex: str | None = None,
    use_calculator: bool = True,
) -> list[tuple[str, int]]:
    """Greedy decoding of many prompts at once, calculator in the loop."""
    state, next_logits, pos_t = prefill_batch(model, [tokenizer.encode(p).ids for p in prompts])
    return decode_batch_greedy(
        model, tokenizer, state, next_logits, pos_t, max_new_tokens, eos_id, stop,
        stop_regex, use_calculator,
    )


@torch.no_grad()
def left_pad(prompts_ids: list[list[int]], device, pad_id: int = 0):
    """(ids (B, T), mask (B, T)) with every prompt right-aligned at column T-1."""
    T = max(len(p) for p in prompts_ids)
    ids = torch.full((len(prompts_ids), T), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(len(prompts_ids), T, dtype=torch.bool, device=device)
    for i, p in enumerate(prompts_ids):
        ids[i, T - len(p):] = torch.tensor(p, device=device)
        mask[i, T - len(p):] = True
    return ids, mask


@torch.no_grad()
def prefill_batch(model, prompts_ids: list[list[int]], chunk_size: int = 256):
    """Prefill prompts of different lengths in one left-padded call.

    Returns (state, next_logits (B, V), positions (B,))."""
    device = next(model.parameters()).device
    ids, mask = left_pad(prompts_ids, device)
    lg, _, state = model(ids, mode="chunkwise", chunk_size=chunk_size, return_state=True, mask=mask)
    pos = torch.full((len(prompts_ids),), ids.shape[1], device=device)
    return state, lg[:, -1], pos


@torch.no_grad()
def decode_batch_greedy(
    model,
    tokenizer,
    state,
    next_logits,
    pos_t,
    max_new_tokens: int = 256,
    eos_id: int | None = 0,
    stop: tuple[str, ...] = (),
    stop_regex: str | None = None,
    use_calculator: bool = True,
) -> list[tuple[str, int]]:
    """Greedy decoding from prefilled states, calculator in the loop."""
    device = next_logits.device
    n = next_logits.shape[0]
    pattern = re.compile(stop_regex) if stop_regex else None
    outs: list[list[int]] = [[] for _ in range(n)]
    queued: list[list[int]] = [[] for _ in range(n)]
    calls, done = [0] * n, [False] * n
    pad = eos_id if eos_id is not None else 0
    for _ in range(max_new_tokens + 64):  # + room for calculator results
        greedy = next_logits.argmax(-1).tolist()
        step = []
        for i in range(n):
            if done[i]:
                step.append(pad)
                continue
            if queued[i]:
                tok = queued[i].pop(0)
            else:
                tok = greedy[i]
                if (eos_id is not None and tok == eos_id) or len(outs[i]) >= max_new_tokens:
                    done[i] = True
                    step.append(pad)
                    continue
            outs[i].append(tok)
            step.append(tok)
            if queued[i]:
                continue  # still writing a calculator result
            text = tokenizer.decode(outs[i])
            if any(text.endswith(s) for s in stop) or (pattern and pattern.search(text)):
                done[i] = True
                continue
            inject = _calculator_tokens(text, tokenizer) if use_calculator else None
            if inject:
                queued[i] = list(inject)
                calls[i] += 1
        if all(done):
            break
        lg, _, state = model(
            torch.tensor(step, device=device)[:, None],
            state=state,
            offset=pos_t,
            mode="recurrent",
        )
        pos_t = pos_t + 1
        next_logits = lg[:, -1]
    return [(tokenizer.decode(o), c) for o, c in zip(outs, calls)]


# --------------------------------------------------------------------------
# Training data: mark correct arithmetic as calculator calls
# --------------------------------------------------------------------------
_RHS = re.compile(r"=(\s*\$?)(-?\d[\d,]*(?:\.\d+)?)(?![\w^_{}\\%<]|\.\d|,\d)")
_LHS_CHARS = set("0123456789$()+-*/×÷ \t")
_BLOCKED_BEFORE = set(".$\\{}^_)<>")
_HAS_OPERATION = re.compile(r"[\d)][-+*/][\d(]")
_GAP = re.compile(r"\d\s+\d")
_MAX_LHS = 200


def _lhs_start(text: str, eq: int) -> int:
    """Index where the arithmetic before text[eq] (an "=") begins."""
    i = eq
    while i > 0 and eq - i < _MAX_LHS:
        c = text[i - 1]
        if c in _LHS_CHARS:
            i -= 1
        elif c in ",." and i >= 2 and text[i - 2].isdigit() and text[i].isdigit():
            i -= 1  # 1,200 or 3.5
        elif c == "x" and i >= 2 and text[i - 2] == " " and text[i] == " ":
            i -= 1  # "4 x 3"
        else:
            break
    return i


def _matches(value: str, written: str) -> bool:
    """Does the calculator's value agree with the number the text wrote?"""
    try:
        v, w = float(value), float(written.replace(",", ""))
    except ValueError:
        return False
    decimals = len(written.split(".", 1)[1]) if "." in written else 0
    tol = 1e-9 if decimals == 0 else 0.5 * 10**-decimals + 1e-9
    return abs(v - w) <= tol


def annotate_arithmetic(text: str) -> tuple[str, int]:
    """Rewrite correct worked arithmetic in the calculator convention:
      "16 * $47 = $752"  ->  "16 * $47 = $<<16*47=752>>752"
    """
    pieces, count, last, prev_end = [], 0, 0, 0
    for m in _RHS.finditer(text):
        eq = m.start()
        lo = max(_lhs_start(text, eq), prev_end)
        prev_end = m.end()
        raw = text[lo:eq]
        body = raw.lstrip()
        if not body or not (body[0].isdigit() or body[0] in "$("):
            continue
        start = lo + len(raw) - len(body)
        if start > 0 and (text[start - 1].isalnum() or text[start - 1] in _BLOCKED_BEFORE):
            continue
        lhs = body.rstrip()
        expr = re.sub(r"[\s$,]", "", lhs.replace(" x ", "*")).replace("×", "*").replace("÷", "/")
        if start < last or not _HAS_OPERATION.search(expr) or _GAP.search(lhs):
            continue
        value = safe_eval(expr)
        if value is None or not _matches(value, m.group(2)):
            continue
        pieces.append(text[last : m.end(1)] + f"<<{expr}={value}>>")
        last = m.end(1)
        count += 1
    pieces.append(text[last:])
    return "".join(pieces), count
