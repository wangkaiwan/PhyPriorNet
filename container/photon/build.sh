#!/bin/bash
# Stage photon weights (NOT in git) + build the photon GC invoke image. Run from repo root:
#   bash container/photon/build.sh [WEIGHTS_PT]
set -eu
cd "$(git rev-parse --show-toplevel)"
W=${1:-/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt}
DST=container/photon/weights
mkdir -p "$DST"
cp "$W" "$DST/photon.pt"
cp /data/kwang/DoseRad2026_raw/beam_parameters.json "$DST/beam_parameters.json"
echo "[build] staged $(du -h $DST/photon.pt | cut -f1) photon.pt"
docker build -f container/photon/Dockerfile -t doserad-photon:latest .
echo "[build] done -> doserad-photon:latest"
