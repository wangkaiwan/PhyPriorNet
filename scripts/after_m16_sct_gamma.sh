#!/bin/bash
# When the margin-16 pipeline finishes (frees GPU1), run the HELD-OUT sCT dose-gamma test: does the NEW
# whole-image sCT (refiner@165) beat the OLD sCT on the 6 web-test pairs, in dose space? Same production
# proton dose net on real-CT vs old-sCT vs new-sCT density -> engine error cancels, only sCT differs.
# This is the TRUSTWORTHY proton-MRI read (held16 in-sample is fantasy) and decides if Step 4 is worth
# finishing (~17h left). Autonomous (user on vacation).
set -u
cd /home/kaiwang/project/DoseRad2026
PLOG=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/b9tt5b4sd.output
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough|out\[idx"'
echo "[sct-gamma] waiting for margin-16 pipeline to free GPU1 ..."
until grep -q "M16 PIPELINE DONE" "$PLOG" 2>/dev/null || [ -z "$(pgrep -f after_m16_traineval)" ]; do sleep 120; done
sleep 20
echo "[sct-gamma] GPU1 free -> held-out sCT dose gamma: NEW whole-image sCT (ref@165) vs OLD sCT (6 web-test)"
DOSERAD_WHOLE_REF=1 \
DOSERAD_NEW_REF=/data/kwang/sct_refine_runs/ref_1x1x3_samefield_whole2/best.pt \
DOSERAD_NEW_CLF=/data/kwang/sct_classify_runs/clf_1x1x3_samefield_whole/best.pt \
CUDA_VISIBLE_DEVICES=1 $CR scripts/eval_sct_gamma_test.py 2>&1 | eval $F | grep -iE "new sCT|MEAN|gamma|old|new|real|patient|Error|Traceback" | tail -30
echo "[sct-gamma] SCT GAMMA TEST DONE — if new<=old, whole-image sCT did NOT help -> consider stopping Step 4"
