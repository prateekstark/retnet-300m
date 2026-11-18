"""Latent reasoning ("continuous thought", Coconut-style) on RetNet.

After the question, the model enters latent mode with `<bot>`. Instead of
writing reasoning tokens, it feeds its own last hidden state back in as the
next input embedding for `n_latent` steps, then `<eot>` switches back to text
and it writes the rest of the solution and the answer.

    Question: ...\\nAnswer:<bot> [h1][h2]...[hn] <eot> remaining steps\\n#### 42<eos>

Why RetNet suits this: every latent step is one recurrent update of a
fixed-size state, so a thought costs the same memory at step 1000 as at step 1
(a Transformer's KV cache grows with every thought).

Training follows the Coconut curriculum (Hao et al., 2024): stage 0 is plain
step-by-step SFT; at stage s the first s reasoning steps are replaced by
s * c latent thoughts. Gradients flow through every thought (backprop through
the recurrence), so the thoughts learn to carry what the removed steps said.

The thought input is `scale * norm_f(h)`: `norm_f(h)` is what the LM head
reads, and a learned scalar `scale` (initialized to match the token-embedding
norm) keeps it on the same footing as a real token embedding in the residual
stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .calculator import _cat_states, annotate_arithmetic, decode_batch_greedy

BOT = "<|reserved_0|>"  # begin latent thought
EOT = "<|reserved_1|>"  # end latent thought
EOS = "<|endoftext|>"


@dataclass(frozen=True)
class Problem:
    question: str
    steps: tuple[str, ...]  # reasoning lines, calculator-annotated
    answer: str  # final number as written after '####'


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def parse_gsm8k(question: str, answer: str) -> Problem | None:
    """GSM8K rows: steps on separate lines, then '#### <answer>'."""
    if "####" not in answer:
        return None
    body, final = answer.rsplit("####", 1)
    steps = tuple(s.strip() for s in body.strip().split("\n") if s.strip())
    final = final.strip().replace(",", "")
    if not steps or not final:
        return None
    return Problem(question.strip(), steps, final)


_METAMATH_ANSWER = re.compile(r"The answer is:?\s*(.+?)\s*$", re.S)


def parse_metamath(query: str, response: str) -> Problem | None:
    """MetaMathQA rows: free-form steps, then 'The answer is: <answer>'.
    Arithmetic is rewritten as calculator calls (<<a*b=c>>c) like pretraining."""
    m = _METAMATH_ANSWER.search(response)
    if not m:
        return None
    final = m.group(1).strip().rstrip(".").replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", final):
        return None  # keep numeric answers only (GSM-style)
    body = response[: m.start()].strip()
    steps = []
    for line in body.split("\n"):
        line = line.strip()
        if line:
            steps.append(annotate_arithmetic(line)[0])
    if not steps:
        return None
    return Problem(query.strip(), tuple(steps), final)


def prompt_text(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def suffix_text(p: Problem, n_removed: int) -> str:
    rest = list(p.steps[n_removed:]) + [f"#### {p.answer}"]
    return " " + "\n".join(rest)


@dataclass
class Encoded:
    prefix: list[int]  # prompt + <bot>
    suffix: list[int]  # <eot> + remaining steps + answer + <eos>


def encode(tok, p: Problem, n_removed: int, ids: dict[str, int]) -> Encoded:
    prefix = tok.encode(prompt_text(p.question)).ids + [ids[BOT]]
    suffix = [ids[EOT]] + tok.encode(suffix_text(p, n_removed)).ids + [ids[EOS]]
    return Encoded(prefix, suffix)


def special_ids(tok) -> dict[str, int]:
    out = {}
    for name in (BOT, EOT, EOS):
        i = tok.token_to_id(name)
        if i is None:
            raise ValueError(
                f"tokenizer has no {name}; latent reasoning needs a tokenizer with "
                "reserved tokens (prepare_mix.py v0.5+)"
            )
        out[name] = i
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class LatentReasoner(nn.Module):
    """Wraps a RetNetLM with the thought feedback path."""

    def __init__(self, model: nn.Module, scale: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    @torch.no_grad()
    def init_scale(self, prefixes: list[list[int]], chunk_size: int = 128) -> float:
        """Set `scale` so thought inputs have the norm of real token embeddings."""
        m = self.model
        dev = m.embed.weight.device
        emb, hid = [], []
        for ids in prefixes:
            t = torch.tensor([ids], device=dev)
            x = m.embed(t)
            h, _, _ = m.trunk(x, mode="chunkwise", chunk_size=chunk_size, return_state=True)
            emb.append(x[0].float().norm(dim=-1).mean())
            hid.append(m.norm_f(h[0, -1]).float().norm())
        s = float(torch.stack(emb).mean() / torch.stack(hid).mean())
        self.scale.fill_(s)
        return s

    def thought_input(self, h: Tensor) -> Tensor:
        return (self.scale * self.model.norm_f(h)).to(h.dtype)

    def prefill(self, prefixes: list[list[int]], chunk_size: int = 128):
        """Each prefix on its own (lengths differ), states stacked.

        Returns (last hidden (B,1,d), state, positions (B,))."""
        m = self.model
        dev = m.embed.weight.device
        states, last = [], []
        for ids in prefixes:
            x = m.embed(torch.tensor([ids], device=dev))
            h, st, _ = m.trunk(x, mode="chunkwise", chunk_size=chunk_size, return_state=True)
            states.append(st)
            last.append(h[:, -1:])
        pos = torch.tensor([len(p) for p in prefixes], device=dev)
        return torch.cat(last, 0), _cat_states(states), pos

    def think(self, h: Tensor, state, pos: Tensor, n_latent: int):
        """Feed the last hidden state back as input for n_latent recurrent steps."""
        for _ in range(n_latent):
            h, state, _ = self.model.trunk(
                self.thought_input(h), state, offset=pos, mode="recurrent"
            )
            pos = pos + 1
        return h, state, pos

    def loss(self, batch: list[Encoded], n_latent: int, chunk_size: int = 128) -> Tensor:
        """Cross-entropy on everything after <eot>, backprop through all thoughts."""
        m = self.model
        dev = m.embed.weight.device
        h, state, pos = self.prefill([b.prefix for b in batch], chunk_size)
        h, state, pos = self.think(h, state, pos, n_latent)
        T = max(len(b.suffix) for b in batch)
        ids = torch.zeros(len(batch), T, dtype=torch.long, device=dev)
        tgt = torch.full((len(batch), T), -1, dtype=torch.long, device=dev)
        for i, b in enumerate(batch):
            s = torch.tensor(b.suffix, device=dev)
            ids[i, : len(s)] = s
            tgt[i, : len(s) - 1] = s[1:]
        x, _, _ = m.trunk(m.embed(ids), state, offset=pos, mode="chunkwise", chunk_size=chunk_size)
        logits = m.lm_head(m.norm_f(x))
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1)

    @torch.no_grad()
    def generate(
        self,
        tok,
        questions: list[str],
        n_latent: int,
        max_new_tokens: int = 256,
        use_calculator: bool = True,
        stop_regex: str | None = r"####[^\n]*\d[^\n]*(\n|$)",
    ) -> list[tuple[str, int]]:
        """Think for n_latent steps, emit <eot>, then decode greedily with the calculator."""
        ids = special_ids(tok)
        dev = self.model.embed.weight.device
        prefixes = [tok.encode(prompt_text(q)).ids + [ids[BOT]] for q in questions]
        h, state, pos = self.prefill(prefixes)
        h, state, pos = self.think(h, state, pos, n_latent)
        eot = torch.full((len(questions), 1), ids[EOT], device=dev)
        lg, _, state = self.model(eot, state=state, offset=pos, mode="recurrent")
        return decode_batch_greedy(
            self.model, tok, state, lg[:, -1], pos + 1, max_new_tokens,
            eos_id=ids[EOS], stop=("\nQuestion:",), stop_regex=stop_regex,
            use_calculator=use_calculator,
        )


def answer_of(text: str) -> str | None:
    """Number after '####' (normalized), else None."""
    if "####" not in text:
        return None
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text.split("####", 1)[1])
    if not m:
        return None
    n = m.group(0).replace(",", "")
    try:
        f = float(n)
    except ValueError:
        return None
    return str(int(f)) if f.is_integer() else str(f)


def normalize_answer(a: str) -> str | None:
    return answer_of(f"#### {a}")
