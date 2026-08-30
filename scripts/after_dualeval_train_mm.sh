#!/bin/bash
set -u
cd /home/kaiwang/project/DoseRad2026
DUAL=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/b3alk9gjb.output
echo "[mm] waiting for photon-MRI dual eval to finish ..."
until grep -q "DUAL EVAL DONE" "$DUAL" 2>/dev/null; do sleep 90; done
sleep 20
echo "[mm] dual eval done -> launching ② multi-modal (ct_mix 0.6, margin-8) on GPU1"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 conda run -n doserad --no-capture-output python -u \
  scripts/train_dose_e2e.py --config configs/experiments/all75/mm_ftm8_photonmri.yaml 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough" | grep -E "init|ct-mix|step (1|[0-9]000)/|val|best|done|Error|Traceback" | tail -50
echo "[mm] ② MULTI-MODAL TRAINING DONE"
