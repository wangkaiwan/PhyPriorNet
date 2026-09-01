#!/bin/bash
# Watch the proton-MRI 1x1x3 E2E: when state.pt reaches step 30000 then 40000, archive the
# snapshot (so the next ckpt can't clobber it) and run the corrected held16 gamma eval on GPU1.
# Fulfills the "eval held16 every 10k, keep the best" agreement.
set -u
cd /home/kaiwang/project/DoseRad2026
RUN=/home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_protonmri
CR="conda run -n doserad --no-capture-output python -u"
step_of(){ $CR -c "import torch;print(torch.load('$RUN/state.pt',map_location='cpu').get('step'))" 2>/dev/null | tail -1; }

for TARGET in 30000 40000; do
  echo "[watch] waiting for step $TARGET ..."
  while :; do
    S=$(step_of)
    [ "$S" = "$TARGET" ] && break
    sleep 120
  done
  K=$RUN/ckpt_$((TARGET/1000))k.pt
  cp $RUN/state.pt $K
  echo "[watch] step $TARGET reached -> archived $K, evaluating held16 ..."
  CUDA_VISIBLE_DEVICES=1 $CR scripts/eval_proton_e2e_held16.py $K 2>&1 \
    | grep -vE "Warning|warn|win_data|out\[idx" | grep -E ">>>|gamma1/1"
done
echo "[watch] DONE 30k + 40k evaluated"
