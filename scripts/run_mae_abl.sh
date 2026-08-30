#!/bin/bash
# Photon-MRI Beam-MAE↔gamma ablation sweep (freeze synth, finetune dose net 5k steps from deployed
# m24S2_p4_mmB with reduced grad/het/lung). Eval each on held16[:8] for Beam MAE + gamma. GPU1.
set -u
cd /home/kaiwang/project/DoseRad2026
export WANDB_MODE=disabled
CR="conda run -n doserad --no-capture-output python -u"
NEVAL=8
FILT='grep -vE "Warning|warn|win_data|out\[idx|GradScaler|scaler =|Deprecat|Not enough"'

echo "############ A0 baseline (deployed m24S2_p4_mmB) ############"
DOSERAD_NEVAL=$NEVAL CUDA_VISIBLE_DEVICES=1 $CR scripts/diag_photon_mae.py 2>&1 \
  | grep -vE "Warning|warn|win_data|out.idx|Deprecat|Not enough" | grep -E ">>>|Stratified|high\(|mid\(|low\(|beam-max:"

for A in A1 A2 A3; do
  echo "############ $A train (5k) ############"
  CUDA_VISIBLE_DEVICES=1 $CR scripts/train_dose_e2e.py --config configs/experiments/all75/mae_abl_$A.yaml 2>&1 \
    | grep -vE "Warning|warn|win_data|out.idx|GradScaler|scaler =|Deprecat|Not enough" | grep -E "freeze|step (100|2500|5000)/|best|saved" | tail -4
  echo "############ $A eval ############"
  DOSERAD_W_OVERRIDE=/home/kaiwang/doserad2026_workdir/runs/mae_abl_$A/state.pt DOSERAD_NEVAL=$NEVAL \
    CUDA_VISIBLE_DEVICES=1 $CR scripts/diag_photon_mae.py 2>&1 \
    | grep -vE "Warning|warn|win_data|out.idx|Deprecat|Not enough" | grep -E ">>>|Stratified|high\(|mid\(|low\("
done
echo "############ SWEEP DONE ############"
