#!/bin/bash
# Apples-to-apples held16 proton-CT gamma: single-Gaussian champion (all75_r2_ft) vs double-Gaussian halo
# (all75_r2_ft_double, DOSERAD_LATERAL_DOUBLE=1 so build_ray uses the double prior at inference == training).
# Same script/cohort. Watch ALL/abd/lung — prior-only showed lung slightly negative. GPU1.
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough|out\[idx"'
echo "########## SINGLE-Gaussian champion (all75_r2_ft) — baseline (expect ~97.5) ##########"
CUDA_VISIBLE_DEVICES=1 $CR scripts/eval_deploy_held16.py proton_ct 2>&1 | eval $F | grep -iE "ALL|abd|lung|gamma|>>>|mean" | tail -8
echo "########## DOUBLE-Gaussian halo (all75_r2_ft_double + LATERAL_DOUBLE=1) ##########"
DOSERAD_W_OVERRIDE=/home/kaiwang/doserad2026_workdir/runs/all75_r2_ft_double/best.pt \
DOSERAD_LATERAL_DOUBLE=1 \
CUDA_VISIBLE_DEVICES=1 $CR scripts/eval_deploy_held16.py proton_ct 2>&1 | eval $F | grep -iE "ALL|abd|lung|gamma|>>>|mean" | tail -8
echo "DOUBLE-VS-SINGLE EVAL DONE"
