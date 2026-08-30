#!/bin/bash
# When the double-Gaussian retrain frees GPU1, RESUME Step 4 (proton-MRI whole-sCT dose-aware E2E) from its
# state.pt (~step 25k) to 40k. It was killed on a BUGGY sCT test (coarse mismatch); the whole-image sCT is
# actually ON-PAR with old (94.8 vs 95.4). User: "空了恢复它". --resume loads model/opt/sched/step (max_steps
# unchanged at 40k, so no OneCycle raise-crash). Autonomous.
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough|out\[idx"'
DBLOG=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/bg8190g0w.output

echo "[resume-s4] waiting for the double-Gaussian retrain to free GPU1 ..."
until grep -qE "DOUBLE-GAUSSIAN RETRAIN DONE|ABORT" "$DBLOG" 2>/dev/null || [ -z "$(pgrep -f after_confirm_double_retrain)" ]; do sleep 300; done
sleep 30
echo "[resume-s4] GPU1 free -> resuming Step 4 (proton-MRI whole-sCT E2E) from state.pt to 40k"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 $CR scripts/train_dose_proton_e2e.py \
  --config configs/experiments/all75/e2e_1x1x3_whole_protonmri.yaml --resume 2>&1 \
  | eval $F | grep -E "resume|init|warm|step (1|[0-9]0000)/|val|best|done|Error|Traceback" | tail -40
echo "[resume-s4] STEP4 RESUME DONE -> eval the E2E synth with the CORRECTED whole-image sCT gamma test next"
