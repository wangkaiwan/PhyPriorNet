#!/bin/bash
# RESUME a paused foldexp (arm A DONE = teacher; arm B interrupted). Runs B(--resume) → D → C → eval
# on the given GPU. Usage: bash scripts/run_foldexp_resume.sh <photon|proton> <GPU> [STEPS_BC]
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CFG=configs/experiments/foldexp
LOGS=/home/kaiwang/doserad2026_workdir/logs
MOD=$1; G=$2; STEPS=${3:-80000}
if [ "$MOD" = "photon" ]; then
  TRAIN=scripts/distill_dose_photon.py; PFX=ct_f1; LPFX=foldexp; EVAL="scripts/eval_foldexp_photonct.sh"
else
  TRAIN=scripts/train_dose_proton.py;   PFX=pct_f1; LPFX=foldexp_pct; EVAL="scripts/eval_foldexp_protonct.sh"
fi
DRV=$LOGS/foldexp_${MOD}ct_resume_driver.log
echo "[resume $MOD] GPU=$G STEPS=$STEPS  ($(date '+%H:%M' 2>/dev/null || echo now))"

echo "[resume $MOD] STAGE B (--resume from state.pt)"
CUDA_VISIBLE_DEVICES=$G $CR $TRAIN --config $CFG/${PFX}_B_base32_gtkd.yaml --max-steps $STEPS --resume \
  >> $LOGS/${LPFX}_B_base32gtkd.log 2>&1
echo "[resume $MOD] STAGE D (init<-B + gt-ft)"
CUDA_VISIBLE_DEVICES=$G $CR $TRAIN --config $CFG/${PFX}_D_base32_distillinit_ft.yaml \
  > $LOGS/${LPFX}_D_base32ftgt.log 2>&1
echo "[resume $MOD] STAGE C (base32 scratch)"
CUDA_VISIBLE_DEVICES=$G $CR $TRAIN --config $CFG/${PFX}_C_base32_scratch.yaml --max-steps $STEPS \
  > $LOGS/${LPFX}_C_base32scratch.log 2>&1
echo "[resume $MOD] ALL TRAINING DONE — eval on fold_1 held-out"
bash $EVAL "$G"
echo "[resume $MOD] COMPLETE"
