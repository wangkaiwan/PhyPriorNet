#!/bin/bash
# When Step 4 (proton-MRI 40k whole-image sCT E2E) finishes, eval it on held16 with the WHOLE coarse
# (in-sample screening vs baseline 94.8). GPU0 frees when Step 4 ends. Autonomous (user on vacation).
set -u
cd /home/kaiwang/project/DoseRad2026
S4LOG=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/byes4knp6.output
CKPT=/home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_whole_protonmri/best.pt
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough|out\[idx"'
echo "[s4eval] waiting for Step 4 to finish ..."
until grep -qE "STEP4 40K DOSE E2E DONE|STEP4 DONE" "$S4LOG" 2>/dev/null || [ -z "$(pgrep -f 'train_dose_proton_e2e.*whole_protonmri')" ]; do sleep 120; done
sleep 20
[ -f "$CKPT" ] || { echo "[s4eval] no best.pt — Step 4 may have crashed"; ls -la /home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_whole_protonmri/ 2>/dev/null; exit 1; }
echo "[s4eval] Step 4 done -> held16 eval (whole coarse) on GPU0"
DOSERAD_E2E_CFG=configs/experiments/all75/e2e_1x1x3_whole_protonmri.yaml \
DOSERAD_COARSE=/data/kwang/coarse_ct_1x1x3_samefield_whole_soft \
CUDA_VISIBLE_DEVICES=0 $CR scripts/eval_proton_e2e_held16.py "$CKPT" 2>&1 | eval $F | grep -E ">>>|gamma|Error|Traceback" | tail -25
echo "[s4eval] STEP4 EVAL DONE (vs baseline 94.8 in-sample; leaderboard held-out is the real arbiter)"
