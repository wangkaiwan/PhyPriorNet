#!/bin/bash
# Distill a base32 photon-CT student from the CURRENT leaderboard champion 4018f597 (docker_extracted/
# photon_ct_docker.pt), full-data, on the m24 cache. Goal: 1.55x-faster net (compute-bound photon) at
# teacher-parity gamma -> photon-CT runtime -> 1st. accel/docs V8 proved base32 = teacher (lossless).
set -u
cd /home/kaiwang/project/DoseRad2026
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 conda run -n doserad --no-capture-output python -u \
  scripts/distill_dose_photon.py --config configs/experiments/all75/distill_photonct_b32_from4018.yaml 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough" \
  | grep -E "teacher|student|init|step|loss|val|best|saved|done|Error|Traceback" | tail -80
echo "DISTILL FROM4018 DONE"
