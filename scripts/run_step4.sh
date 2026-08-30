#!/bin/bash
# Step 4 of the proton whole-image sCT chain: 40k proton-MRI dose E2E on GPU0, warm-started from the
# new whole-image refiner (synth_ckpt @ ep165) + whole coarse + all75_r2_ft dose net. freeze_synth:false
# -> the 1x1x3 sCT becomes dose-aware. Held16 eval is set up separately (needs the WHOLE coarse dir).
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough|out\[idx"'
echo "### Step 4: proton-MRI 40k dose E2E (whole-image sCT @165) — GPU0 ###"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 $CR scripts/train_dose_proton_e2e.py \
  --config configs/experiments/all75/e2e_1x1x3_whole_protonmri.yaml 2>&1 \
  | eval $F | grep -E "warm-start|init|step (1|[0-9]0000)/|val|best|saved|done|Error|Traceback" | tail -50
echo "### STEP4 40K DOSE E2E DONE ###"
ls -la --time-style=long-iso /home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_whole_protonmri/*.pt 2>/dev/null
