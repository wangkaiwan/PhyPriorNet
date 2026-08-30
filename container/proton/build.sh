#!/bin/bash
# Stage weights (NOT in git) + build the proton GC invoke image. Run from repo root:
#   bash container/proton/build.sh [WEIGHTS_PT]
set -eu
cd "$(git rev-parse --show-toplevel)"
W=${1:-/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt}
DST=container/proton/weights
mkdir -p "$DST"
cp "$W" "$DST/proton.pt"
cp /data/kwang/DoseRad2026_raw/beam_parameters.json "$DST/beam_parameters.json"
# proton machine kernel npz (ProtonMachineData default) — bake it too if not already in doserad/
echo "[build] staged weights: $(du -h $DST/proton.pt | cut -f1) proton.pt"
docker build -f container/proton/Dockerfile -t doserad-proton:latest .
echo "[build] done -> doserad-proton:latest"
