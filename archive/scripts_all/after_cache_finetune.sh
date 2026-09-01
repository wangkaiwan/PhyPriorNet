#!/bin/bash
# Wait for the margin-8 cache build to finish, then launch the margin-8 finetune of the best photon-CT
# (4018f597) on GPU1. Checkpoints every 5000 steps -> pick best by held16 later.
set -u
cd /home/kaiwang/project/DoseRad2026
CACHE_LOG=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/b2040g8s0.output
M8=/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m8
echo "[ft] waiting for margin-8 cache build ..."
until grep -q "M8 CACHE BUILD DONE" "$CACHE_LOG" 2>/dev/null; do sleep 120; done
if [ ! -d "$M8" ] || [ "$(find "$M8" -maxdepth 1 -type d 2>/dev/null | wc -l)" -lt 60 ]; then
  echo "[ft] ABORT: margin-8 cache incomplete ($M8) — check the build log"; exit 1
fi
echo "[ft] cache ready ($(find "$M8" -maxdepth 1 -type d | wc -l) pt-dirs) -> launching margin-8 finetune"
sleep 20
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 conda run -n doserad --no-capture-output python -u \
  scripts/train.py --config configs/experiments/all75/ftm8_from_best.yaml 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough" | grep -E "init|step|loss|val|best|saved|done|Error|Traceback" | tail -60
echo "[ft] margin-8 finetune DONE"
