#!/bin/bash
# Build the margin-8 photon crop cache for the margin-8 finetune (photon_skinentry_ssd was deleted).
# Step 1: dose crops + GT at margin-8 -> photon_dose_m16   Step 2: add physics channels -> photon_skinentry_m16
set -eu
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
FILT='grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough"'
echo "########## STEP 1: build_photon_targets --margin 16 (photon_dose_m16) ##########"
CUDA_VISIBLE_DEVICES=1 $CR scripts/build_photon_targets.py --margin 16 --shard 0/1 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough" | grep -E "done|/|pid|Error|Traceback" | tail -20
echo "########## STEP 2: build_photon_cache_m24 --margin 16 (photon_skinentry_m16) ##########"
CUDA_VISIBLE_DEVICES=1 $CR scripts/build_photon_cache_m24.py --margin 16 --shard 0/1 \
  --src /data/kwang/doserad_cache/photon_dose_m16 \
  --out /home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m16 2>&1 \
  | grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough" | grep -E "done|/|Error|Traceback" | tail -20
echo "########## M16 CACHE BUILD DONE ##########"
ls -d /home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m16 2>/dev/null && echo "cache present"
