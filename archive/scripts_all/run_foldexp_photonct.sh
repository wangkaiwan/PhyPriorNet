#!/bin/bash
# FOLD EXPERIMENT driver (user 2026-08-21): base32 direct-train vs distilled, on a REAL held-out fold.
# DAG:  [A base48 teacher] ∥ [C base32 scratch]   → B (base32 gt+kd, needs A) → D (base32 init<-B + gt-ft)
# Then eval all 4 on fold_1's 15 held-out (cv_eval_photonct_full, margin-24, GT full-grid plans).
# Usage:  bash scripts/run_foldexp_photonct.sh [GPU_chain] [GPU_parallelC] [STEPS_ABC]
#   GPU_chain    = GPU for the serial A→B→D chain           (default 0)
#   GPU_parallelC= GPU for C (runs alongside A)             (default 1)
#   STEPS_ABC    = max_steps for A/B/C (D uses its own 40k) (default 80000 = the locked SCREEN budget)
# If you only have ONE GPU, pass the same index for both (A and C then run serially on it).
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CFG=configs/experiments/foldexp
LOGS=/home/kaiwang/doserad2026_workdir/logs
mkdir -p "$LOGS"
GA=${1:-0}; GC=${2:-1}; STEPS=${3:-80000}   # locked: 80k screen (user 2026-08-21)
echo "[foldexp] chain(A,B,D) GPU=$GA | C GPU=$GC | STEPS(A,B,C)=$STEPS | D=40k(its config)"

echo "[foldexp] STAGE 1: A(base48 teacher) on GPU$GA  +  C(base32 scratch) on GPU$GC  (parallel)"
CUDA_VISIBLE_DEVICES=$GA $CR scripts/distill_dose_photon.py \
  --config $CFG/ct_f1_A_base48_teacher.yaml --max-steps $STEPS > "$LOGS/foldexp_A_base48.log" 2>&1 &
PA=$!
if [ "$GC" != "$GA" ]; then
  CUDA_VISIBLE_DEVICES=$GC $CR scripts/distill_dose_photon.py \
    --config $CFG/ct_f1_C_base32_scratch.yaml --max-steps $STEPS > "$LOGS/foldexp_C_base32scratch.log" 2>&1 &
  PC=$!
else
  PC=""   # single-GPU: run C after the chain (below)
fi

wait $PA
echo "[foldexp] A done. STAGE 2: B(base32 gt+kd, teacher=A) on GPU$GA"
CUDA_VISIBLE_DEVICES=$GA $CR scripts/distill_dose_photon.py \
  --config $CFG/ct_f1_B_base32_gtkd.yaml --max-steps $STEPS > "$LOGS/foldexp_B_base32gtkd.log" 2>&1
echo "[foldexp] B done. STAGE 3: D(base32 init<-B + gt-ft) on GPU$GA"
CUDA_VISIBLE_DEVICES=$GA $CR scripts/distill_dose_photon.py \
  --config $CFG/ct_f1_D_base32_distillinit_ft.yaml > "$LOGS/foldexp_D_base32ftgt.log" 2>&1

if [ -z "$PC" ]; then
  echo "[foldexp] single-GPU: STAGE 1b: C(base32 scratch) on GPU$GA"
  CUDA_VISIBLE_DEVICES=$GA $CR scripts/distill_dose_photon.py \
    --config $CFG/ct_f1_C_base32_scratch.yaml --max-steps $STEPS > "$LOGS/foldexp_C_base32scratch.log" 2>&1
else
  wait $PC; echo "[foldexp] C done."
fi

echo "[foldexp] ALL TRAINING DONE. Running eval on fold_1 held-out (15) ..."
bash scripts/eval_foldexp_photonct.sh "$GA"
echo "[foldexp] COMPLETE. Table above; CSVs in $LOGS/foldexp_*.csv"
