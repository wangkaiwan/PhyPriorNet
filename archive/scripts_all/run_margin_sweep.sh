#!/bin/bash
# Photon margin sweep: for photon_ct + photon_mri, DOSERAD_PHOTON_MARGIN in {24,16,12,8}, held16[:N].
# Reports gamma / Beam MAE / mean crop voxels / predict-time per margin. margin 24 = current docker.
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_margin_sweep.sh [N]   (N default 8; use 16 to confirm)
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
N=${1:-8}
FILT='grep -vE "win_data|out\[idx|Warning|warn|GradScaler|Deprecat|Not enough"'
for TASK in photon_ct photon_mri; do
  echo "############################## $TASK ##############################"
  for M in 24 16 12 8; do
    echo "----- $TASK margin=$M -----"
    DOSERAD_PHOTON_MARGIN=$M $CR scripts/margin_sweep.py $TASK $N 2>&1 \
      | grep -vE "win_data|out\[idx|Warning|warn|GradScaler|Deprecat|Not enough" | grep -E ">>>|gamma "
  done
done
echo "############ MARGIN SWEEP DONE ############"
