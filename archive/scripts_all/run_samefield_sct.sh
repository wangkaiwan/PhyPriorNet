#!/bin/bash
# CONTROL arm for the all-field sCT experiment (user 2026-08-20): BOTH steps same-field (0.35T = competition).
# Model A = clf_whole_samefield_aug (same-field clf) coarse + refiner on the 166 same-field (0.35T) patients.
# Compare vs Model B (both all-field: clf_whole + coarse_allfield + ref_allfield_wt2). Same manifest
# (sct_data_2mm.json) + same val as B; the ONLY differences are the clf (same-field vs all-field) and the
# refiner data (--allfield off vs on). Clean isolation of "same-field vs all-field, both steps".
# Runs on GPU1 (set CUDA_VISIBLE_DEVICES before calling). GPU0 = base40 distillation.
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CLF=/data/kwang/sct_classify_runs/clf_whole_samefield_aug/best.pt
COARSE=/data/kwang/doserad_cache_hdd/coarse_ct_whole_soft_samefield_ctrl
REFOUT=/data/kwang/sct_refine_runs/ref_samefield_ctrl
mkdir -p "$COARSE"

echo "[samefield-sct] STEP 1: regen coarse (clf_whole_samefield_aug, whole-image, soft, 166 sf +val) -> $COARSE"
$CR scripts/precompute_coarse_ct.py --clf "$CLF" --out "$COARSE" --whole-image --soft \
  --data /home/kaiwang/doserad2026_workdir/sct_data_2mm.json 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart" | grep -E "done|[0-9]+/[0-9]+|Error|Traceback" | tail -8
echo "[samefield-sct] coarse count: $(ls $COARSE/*.nii.gz 2>/dev/null | wc -l) (expect ~182 = 166 sf + 16 val)"

echo "[samefield-sct] STEP 2: retrain refiner (0.35T only, NO --allfield) -> $REFOUT"
$CR scripts/train_sct_refiner.py --out "$REFOUT" --coarse-dir "$COARSE" --workers 8 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart" | grep -E "refiner train|ep [0-9]+|best|HU-MAE|Error|Traceback" | tail -12
echo "[samefield-sct] DONE -> $REFOUT/best.pt (Model A; compare vs Model B ref_allfield_wt2 in dose gamma)"
