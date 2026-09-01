#!/bin/bash
set -u
cd /home/kaiwang/project/DoseRad2026
REFDIR=/data/kwang/sct_refine_runs/ref_1x1x3_samefield_whole2
RLOG=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/bxx96l0ne.output
echo "[step4] waiting for whole-image refiner to finish (synth_ckpt.pt) ..."
until [ -f "$REFDIR/synth_ckpt.pt" ] && grep -qE "ep 200/200|done" "$RLOG" 2>/dev/null; do sleep 180; done
sleep 30
echo "[step4] refiner done -> launching 40k dose E2E (whole-image sCT) on GPU0"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 conda run -n doserad --no-capture-output python -u \
  scripts/train_dose_proton_e2e.py --config configs/experiments/all75/e2e_1x1x3_whole_protonmri.yaml 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough" | grep -E "init|warm|step (1|[0-9]0000)/|val|best|done|Error|Traceback" | tail -40
echo "[step4] STEP4 40K DOSE E2E DONE"
