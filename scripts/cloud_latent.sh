#!/bin/bash
# Run the latent-reasoning experiment on a rented GPU (e.g. Vast.ai RTX 5090).
#
# On your machine, copy the pretrained weights to the box (1.4 GB, weights only;
# the optimizer is reset at every curriculum stage anyway):
#   rsync -avP -e "ssh -p PORT" runs/retnet-300m/{ckpt_step_0015082.pt,tokenizer.json,config.json} \
#       root@HOST:/workspace/retnet-300m/runs/retnet-300m/
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
MAX_STAGE=${MAX_STAGE:-4}
C=${C:-2}
METAMATH=${METAMATH:-0}
INIT=${INIT:-runs/retnet-300m}
PY=${PY:-python}

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
$PY -m pip -q install tokenizers datasets
ls "$INIT"/ckpt_step_*.pt "$INIT"/tokenizer.json >/dev/null || {
    echo "missing pretrained weights in $INIT (see the rsync line at the top of this script)"; exit 1; }
$PY -m pytest -q tests/test_latent.py

for MODE in $MODES; do
    OUT=runs/retnet-300m-$MODE
    echo "==================== thought mode: $MODE -> $OUT"
    $PY -u scripts/train_latent.py --init "$INIT" --out "$OUT" --thought_mode "$MODE" \
        --max_stage "$MAX_STAGE" --c "$C" --metamath "$METAMATH" --batch_size 32 --grad_accum 1 \
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
