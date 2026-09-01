#!/bin/bash
# User idea 2026-08-19: sCT REFINER on ALL last-year MRI (every field strength) with the same-field (0.35T =
# this year's competition) UP-WEIGHTED. The CLF already trains on all 341; only the field-sensitive REFINER
# was 0.35T-only. Hypothesis: more domain diversity -> more shift-robust sCT (the MRI leaderboard bottleneck)
# while up-weighting 0.35T keeps it calibrated to the target field. Two steps:
#   1) regen coarse for ALL 341 (clf_whole, whole-image, soft) — currently only 182 (0.35T) exist.
#   2) retrain the refiner with --allfield --samefield_weight 2 on the all-341 coarse.
# Compare (later) the new refiner vs the deployed 0.35T refiner in sCT WEPL / dose gamma.
# Runs on GPU1 (set CUDA_VISIBLE_DEVICES before calling). GPU0 = base40 distillation.
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CLF=/data/kwang/sct_classify_runs/clf_whole/best.pt
COARSE=/data/kwang/doserad_cache_hdd/coarse_ct_whole_soft_allfield
REFOUT=/data/kwang/sct_refine_runs/ref_allfield_wt2
mkdir -p "$COARSE"

echo "[allfield-sct] STEP 1: regen coarse for ALL 341 (clf_whole, whole-image, soft) -> $COARSE"
$CR scripts/precompute_coarse_ct.py --clf "$CLF" --out "$COARSE" --whole-image --soft \
  --data /home/kaiwang/doserad2026_workdir/sct_data_2mm_allpids.json 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart" | grep -E "done|[0-9]+/[0-9]+|Error|Traceback" | tail -20
echo "[allfield-sct] coarse count: $(ls $COARSE/*.nii.gz 2>/dev/null | wc -l) (expect ~341)"

echo "[allfield-sct] STEP 2: retrain refiner --allfield --samefield_weight 2 -> $REFOUT"
$CR scripts/train_sct_refiner.py --out "$REFOUT" --coarse-dir "$COARSE" \
  --allfield --samefield_weight 2 --workers 8 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart" | grep -E "refiner train|ep [0-9]+|best|HU-MAE|Error|Traceback" | tail -40
echo "[allfield-sct] DONE -> $REFOUT/best.pt (eval vs deployed refiner in the morning)"
