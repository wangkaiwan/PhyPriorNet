#!/bin/bash
# FOLD EXPERIMENT driver — PROTON-CT (user 2026-08-21). base32 direct-train vs distilled, on the REAL
# proton-CT held-out fold_1 (valid; proton dose net is shared → answer applies to proton-MRI too, paired
# with the 8/1 best sCT at deploy). Single-GPU serial DAG: A(base48 teacher) → B(base32 gt+kd) →
# D(base32 init<-B + gt-ft) → C(base32 scratch, independent) → eval all 4 on fold_1's 15 held-out.
# Usage: bash scripts/run_foldexp_protonct.sh [GPU] [STEPS_ABC]   (default GPU 1, STEPS 80000; D=40k)
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CFG=configs/experiments/foldexp
LOGS=/home/kaiwang/doserad2026_workdir/logs
mkdir -p "$LOGS"
G=${1:-1}; STEPS=${2:-80000}
echo "[foldexp-proton] GPU=$G | STEPS(A,B,C)=$STEPS | D=40k(its config)"

echo "[foldexp-proton] STAGE A: base48 teacher"
CUDA_VISIBLE_DEVICES=$G $CR scripts/train_dose_proton.py \
  --config $CFG/pct_f1_A_base48_teacher.yaml --max-steps $STEPS > "$LOGS/foldexp_pct_A_base48.log" 2>&1
echo "[foldexp-proton] STAGE B: base32 gt+kd (teacher=A)"
CUDA_VISIBLE_DEVICES=$G $CR scripts/train_dose_proton.py \
  --config $CFG/pct_f1_B_base32_gtkd.yaml --max-steps $STEPS > "$LOGS/foldexp_pct_B_base32gtkd.log" 2>&1
echo "[foldexp-proton] STAGE D: base32 init<-B + gt-ft"
CUDA_VISIBLE_DEVICES=$G $CR scripts/train_dose_proton.py \
  --config $CFG/pct_f1_D_base32_distillinit_ft.yaml > "$LOGS/foldexp_pct_D_base32ftgt.log" 2>&1
echo "[foldexp-proton] STAGE C: base32 scratch"
CUDA_VISIBLE_DEVICES=$G $CR scripts/train_dose_proton.py \
  --config $CFG/pct_f1_C_base32_scratch.yaml --max-steps $STEPS > "$LOGS/foldexp_pct_C_base32scratch.log" 2>&1

echo "[foldexp-proton] ALL TRAINING DONE. Eval on fold_1 held-out (15) ..."
bash scripts/eval_foldexp_protonct.sh "$G"
echo "[foldexp-proton] COMPLETE."
