"""Talk to a trained RetNet-300M reasoning & coding model, with the calculator in the loop.

  python scripts/chat.py --run runs/retnet-300m                       # interactive
  python scripts/chat.py --run runs/retnet-300m --prompt "Question: ... \\nAnswer:"
  python scripts/chat.py --run runs/retnet-300m --code "Write a Python function to merge two sorted lists."
  python scripts/chat.py --run runs/retnet-300m --no_calculator       # ablation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retnet.calculator import generate_with_calculator  # noqa: E402
from retnet.checkpoint import load_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="tokenizer location for older runs")
    ap.add_argument("--prompt", default=None, help="one-shot raw/math prompt")
    ap.add_argument("--code", default=None, help="one-shot coding task prompt")
    ap.add_argument("--lang", default="Python", help="language label for --code")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_calculator", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, tok, step = load_run(a.run, a.device, a.data)
    print(
        f"loaded {a.run} (step {step}, {model.num_params(False) / 1e6:.0f}M params, "
        f"memory={model.cfg.memory}, beta_max={model.cfg.beta_max}, short_conv={model.cfg.short_conv}); "
        f"calculator {'off' if a.no_calculator else 'on'}"
    )

    def answer(prompt: str) -> None:
        t0 = time.time()
        text, calls = generate_with_calculator(
            model,
            tok,
            prompt,
            a.max_new_tokens,
            a.temperature,
            a.top_k,
            stop=("\nQuestion:", "\nProblem:"),
            use_calculator=not a.no_calculator,
        )
        for s in ("\nQuestion:", "\nProblem:"):
            text = text.removesuffix(s)
        print(text)
        print(f"[{calls} calculator call(s), {time.time() - t0:.1f}s]")

    if a.code is not None:
        p = a.code.replace("\\n", "\n")
        answer(f"Problem:\n{p}\n\nSolution ({a.lang}):\n")
        return

    if a.prompt is not None:
        answer(a.prompt.replace("\\n", "\n"))
        return

    print(
        "Interactive mode:\n"
        "  - Type a math/reasoning question (wrapped as 'Question: ...\\nAnswer:')\n"
        "  - Prefix with '/code ' for coding tasks ('Problem:\\n...\\n\\nSolution (Python):')\n"
        "  - Prefix with '>' to send raw text. Ctrl-D to quit."
    )
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith(">"):
            answer(line[1:].lstrip())
        elif line.startswith("/code "):
            task = line[6:].strip()
            answer(f"Problem:\n{task}\n\nSolution ({a.lang}):\n")
        else:
            answer(f"Question: {line}\nAnswer:")


if __name__ == "__main__":
    main()
