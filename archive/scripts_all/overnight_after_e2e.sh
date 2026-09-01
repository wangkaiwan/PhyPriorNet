#!/bin/bash
# Overnight orchestration: wait for the proton-MRI E2E training (pid 1103472) to exit -> GPU0 frees ->
# run the photon margin sweep (CT+MRI, margin 24/16/12/8, held16 full) on GPU0. The 40k held16 eval is
# handled separately by scripts/watch_e2e_held16.sh on GPU1.
set -u
cd /home/kaiwang/project/DoseRad2026
echo "[overnight] waiting for E2E training pid 1103472 to exit (GPU0 free)..."
while [ -d /proc/1103472 ]; do sleep 120; done
echo "[overnight] GPU0 free -> starting margin sweep (N=16)"
sleep 30
CUDA_VISIBLE_DEVICES=0 bash scripts/run_margin_sweep.sh 16
echo "[overnight] margin sweep complete"
