"""Calculator: safe AST evaluator, arithmetic annotation, and batched greedy decoding."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from retnet import RetNetConfig, RetNetLM  # noqa: E402
from retnet.calculator import (  # noqa: E402
    annotate_arithmetic,
    generate_batch_greedy,
    generate_with_calculator,
    safe_eval,
)


@pytest.mark.parametrize(
    "expr,want",
    [
        ("48/2", "24"),
        ("7/3", "2.3333"),
        ("3*4+2", "14"),
        ("(5-8)*2", "-6"),
        ("1,200+300", "1500"),
        ("2**10", "1024"),
        ("10 x 3", "30"),
        ("0.1+0.2", "0.3"),
    ],
)
def test_safe_eval_computes_arithmetic(expr, want):
    assert safe_eval(expr) == want


@pytest.mark.parametrize(
    "expr",
    [
        "1/0",
        "9**999",
        "1e300*1e300",
        "__import__('os')",
        "a+1",
        "(lambda: 1)()",
        "[1,2]",
        "",
    ],
)
def test_safe_eval_rejects_unsafe_or_invalid(expr):
    assert safe_eval(expr) is None


def _tokenizer():
    from prepare_mix import EOS, make_tokenizer
    from tokenizers import pre_tokenizers, trainers

    tok = make_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=[EOS],
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(["Answer: <<48/2=24>> clips."] * 50, trainer)
    return tok


class ScriptedModel(torch.nn.Module):
    def __init__(self, tok, plan="<<48/2=", wrong_answer="9", tail=" clips."):
        super().__init__()
        self.tok, self.plan, self.wrong, self.tail = tok, plan, wrong_answer, tail
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.text = ""

    def _next_piece(self) -> str:
        gen = self.text.split("Answer:", 1)[1]
        if len(gen) < len(self.plan):
            return self.plan[len(gen)]
        after_call = gen[len(self.plan) :]
        if after_call == "":
            return self.wrong
        if ">>" not in after_call:
            return ">>"
        done = gen[gen.index(">>") + 2 :]
        return self.tail[len(done)] if len(done) < len(self.tail) else "<|endoftext|>"

    def forward(self, ids, state=None, offset=0, mode="parallel", chunk_size=64):
        self.text = (self.text if state is not None else "") + self.tok.decode(ids[0].tolist())
        piece = self._next_piece()
        tid = (
            self.tok.token_to_id(piece)
            if piece == "<|endoftext|>"
            else self.tok.encode(piece).ids[0]
        )
        logits = torch.full((1, ids.shape[1], self.tok.get_vocab_size()), -1e9)
        logits[0, -1, tid] = 0.0
        return logits, None, ["state"]


@pytest.mark.parametrize(
    "use_calc,want", [(True, "<<48/2=24>> clips."), (False, "<<48/2=9>> clips.")]
)
def test_calculator_replaces_model_arithmetic(use_calc, want):
    tok = _tokenizer()
    text, calls = generate_with_calculator(
        ScriptedModel(tok),
        tok,
        "Answer:",
        max_new_tokens=40,
        eos_id=tok.token_to_id("<|endoftext|>"),
        use_calculator=use_calc,
    )
    assert text.strip() == want
    assert calls == (1 if use_calc else 0)


@pytest.mark.parametrize("memory,beta_max,short_conv", [("none", 1.0, 0), ("delta", 2.0, 4)])
def test_batch_greedy_matches_one_at_a_time(memory, beta_max, short_conv):
    torch.manual_seed(0)
    tok = _tokenizer()
    cfg = RetNetConfig(
        vocab_size=tok.get_vocab_size(),
        d_model=32,
        n_layers=2,
        n_heads=4,
        memory=memory,
        beta_max=beta_max,
        short_conv=short_conv,
    )
    model = RetNetLM(cfg).double().eval()
    prompts = ["Answer:", "Answer: 48 clips <<", "<<48/2=24>> Answer: Answer: clips."]
    batched = generate_batch_greedy(model, tok, prompts, max_new_tokens=12, eos_id=None)
    single = [generate_with_calculator(model, tok, p, max_new_tokens=12, eos_id=None) for p in prompts]
    assert batched == single


@pytest.mark.parametrize(
    "text,want",
    [
        ("16 * $47 = $752 total", "16 * $47 = $<<16*47=752>>752 total"),
        ("$25 + $15.20 + $6.80 = $47.", "$25 + $15.20 + $6.80 = $<<25+15.20+6.80=47>>47."),
        ("10/3 = 3.33", "10/3 = <<10/3=3.3333>>3.33"),
        ("1,000 + 200 = 1,200", "1,000 + 200 = <<1000+200=1200>>1,200"),
        ("(5+3)*2 = 16", "(5+3)*2 = <<(5+3)*2=16>>16"),
        ("2 x 3 = 6", "2 x 3 = <<2*3=6>>6"),
    ],
)
def test_annotate_marks_correct_arithmetic(text, want):
    assert annotate_arithmetic(text) == (want, 1)
