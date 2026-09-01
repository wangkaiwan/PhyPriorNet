#!/bin/bash
# Wait for the photon-MRI margin sweep to finish (frees GPU1), then eval the 7/24 margin-8-TRAINED
# model all75_p2_ftg @ margin-8 on held16 (official metrics) — to compare margin-8-training vs
# margin-8-inference (4018f597 @ m8, already have: gamma 95.34 / IDD 0.0149 internal).
set -u
cd /home/kaiwang/project/DoseRad2026
SWEEP=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/bsttjqq2l.output
echo "[after] waiting for photon-MRI sweep DONE ..."
until grep -q "############## DONE ##############" "$SWEEP" 2>/dev/null; do sleep 120; done
echo "[after] photon-MRI sweep done -> eval all75_p2_ftg @ margin-8"
sleep 30
DOSERAD_PHOTON_MARGIN=8 \
  DOSERAD_W_OVERRIDE=/home/kaiwang/doserad2026_workdir/runs/all75_p2_ftg/state.pt \
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad --no-capture-output python -u scripts/eval_official_held16.py photon_ct 16 2>&1 \
  | grep -vE "win_data|out\[idx|Warning|warn|FutureWarning|cudart|Deprecat|Not enough" | grep -E ">>>|official-eval|IDD"
echo "[after] all75_p2_ftg @ m8 eval done"
