#!/bin/bash
# Run the latent-reasoning experiment on a rented GPU (e.g. Vast.ai RTX 5090).
#
# Pretrained weights come from the Hugging Face Hub by default
# (INIT=hf://starkprateek1/retnet-300m); INIT can also be a local run dir.
#
# On the box (PyTorch template with CUDA >= 12.8 for RTX 50xx / Blackwell):
#   git clone https://github.com/prateekstark/retnet-300m /workspace/retnet-300m
#   cd /workspace/retnet-300m && tmux new -s latent 'bash scripts/cloud_latent.sh'
#
# Runs the curriculum once per thought mode, then the thought-count sweep on the
# full GSM8K test set. Rerunning the script resumes where it stopped (safe on
# interruptible instances). Copy results back with:
#   rsync -avP -e "ssh -p PORT" --exclude '*.pt' root@HOST:/workspace/retnet-300m/runs/ runs/
set -euo pipefail
cd "$(dirname "$0")/.."

MODES=${MODES:-"latent pause none"}   # latent first: the result that matters
MAX_STAGE=${MAX_STAGE:-3}     # GSM8K-Aug solutions have 2.6 steps on average
C=${C:-2}
DATA=${DATA:-gsm8k_aug}        # gsm8k_aug: 385k equation-step problems (Coconut setup); gsm8k: GSM8K (+ METAMATH)
AUG_LIMIT=${AUG_LIMIT:-0}      # 0 = all GSM8K-Aug problems
COMPILE=${COMPILE:-0}          # 1 = torch.compile the blocks (check scripts/bench_latent.py first)
METAMATH=${METAMATH:-40000}   # extra GSM-style problems: 7.4k GSM8K alone is memorized in one epoch
EPOCHS0=${EPOCHS0:-1}          # stage 0 (explicit step-by-step) epochs
EPOCHS=${EPOCHS:-1}            # epochs per latent stage
INIT=${INIT:-hf://starkprateek1/retnet-300m}   # or a local run dir
PY=${PY:-python}
BATCH=${BATCH:-8}     # 8 x 4 = 32 problems per step; 32 x 1 runs out of memory on 32 GB
ACCUM=${ACCUM:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

$PY - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print(f"GPU {torch.cuda.get_device_name(0)} ({arch}), torch {torch.__version__}, CUDA {torch.version.cuda}")
if arch not in torch.cuda.get_arch_list():
    sys.exit(f"this torch build has no kernels for {arch} ({torch.cuda.get_arch_list()}); "
             "use a PyTorch image built for CUDA 12.8+ (pip install torch --index-url "
             "https://download.pytorch.org/whl/cu128)")
PY
$PY -m pip -q install tokenizers datasets huggingface_hub safetensors pytest
# Fetch (and cache) the pretrained weights now, so a bad INIT fails before training.
$PY - "$INIT" <<'PY'
import sys
sys.path.insert(0, ".")
from retnet.checkpoint import load_run
model, tok, step = load_run(sys.argv[1], "cpu")
print(f"pretrained weights OK: {sys.argv[1]} step {step}")
PY
$PY -m pytest -q tests/test_latent.py

mkdir -p runs
for MODE in $MODES; do
    OUT=runs/retnet-300m-$MODE
    echo "==================== thought mode: $MODE -> $OUT"
    $PY -u scripts/train_latent.py --init "$INIT" --out "$OUT" --thought_mode "$MODE" \
        --max_stage "$MAX_STAGE" --c "$C" --metamath "$METAMATH" --data "$DATA" --aug_limit "$AUG_LIMIT" $([ "$COMPILE" = 1 ] && echo --compile) --epochs_stage0 "$EPOCHS0" --epochs_per_stage "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM" \
        --resume 2>&1 | tee -a "$OUT.train.log"
    if [ ! -f "$OUT/eval_latent.json" ]; then
        $PY -u scripts/eval_latent.py --run "$OUT" \
            --n_latent "$([ "$MODE" = none ] && echo 0 || seq -s, 0 2 $((MAX_STAGE * C + 4)))" 2>&1 | tee -a "$OUT.eval.log"
    fi
done

echo "==================== summary (full GSM8K test)"
$PY - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("runs").glob("retnet-300m-*/eval_latent.json")):
    r = json.loads(p.read_text())
    row = "  ".join(f"{s['n_latent']}:{100 * s['acc']:.1f}" for s in r["sweep"])
    print(f"{r['thought_mode']:7s} (trained {r['trained_n_latent']} thoughts)  thoughts:acc%  {row}")
PY
