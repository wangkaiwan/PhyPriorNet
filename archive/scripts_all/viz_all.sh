#!/usr/bin/env bash
# Per-patient visualization for ALL val patients of one model (run alongside each validation).
# Usage: viz_all.sh <config.yaml> <ckpt> <label> [gpu]
# Outputs viz_<patient>_<label>.png to the workdir; uses the dose/gamma cache (fast on re-run).
set -u
CFG="$1"; CKPT="$2"; LABEL="$3"; GPU="${4:-0}"
PY=/home/kaiwang/.conda/envs/doserad/bin/python
OUT=/home/kaiwang/doserad2026_workdir/viz
mkdir -p "$OUT"
VAL=$($PY -c "import json,yaml; c=yaml.safe_load(open('$CFG')); s=json.load(open(c['splits']))['fold_%d'%c['fold']]['val']; print(' '.join(sorted(s)))")
for p in $VAL; do
  echo "[viz_all] $p ($LABEL)"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
    $PY -u scripts/visualize_case.py --config "$CFG" --ckpt "$CKPT" --patient "$p" \
       --out "$OUT/viz_${p}_${LABEL}.png" 2>&1 | grep -E "wrote|loaded|Error" || true
done
echo "[viz_all] done -> $OUT (viz_*_${LABEL}.png)"
