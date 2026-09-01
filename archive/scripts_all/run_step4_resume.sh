#!/bin/bash
# Resume Step 4 (proton-MRI whole-sCT dose-aware E2E) from state.pt (~step 25k) to 40k on GPU1. It was
# killed on a buggy sCT test (coarse mismatch); the whole-image sCT is actually on-par with old. User asked
# to resume it. --resume loads model/opt/sched/step (max_steps unchanged, no OneCycle raise-crash).
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough|out\[idx"'
echo "[resume-s4] resuming Step 4 (proton-MRI whole-sCT E2E) from state.pt to 40k on GPU1"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 $CR scripts/train_dose_proton_e2e.py \
  --config configs/experiments/all75/e2e_1x1x3_whole_protonmri.yaml --resume 2>&1 \
  | eval $F | grep -E "resume|init|warm|step (1|[0-9]0000)/|val|best|done|Error|Traceback" | tail -40
echo "[resume-s4] STEP4 RESUME DONE"
