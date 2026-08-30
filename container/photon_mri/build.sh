#!/bin/bash
# Stage photon-MRI weights (E2E synth+dose + sCT classifier, NOT in git) + deploy config + machine,
# then build the GC invoke image. Run from repo root:
#   bash container/photon_mri/build.sh [E2E_STATE_PT] [CLF_PT]
set -eu
cd "$(git rev-parse --show-toplevel)"
W=${1:-/home/kaiwang/doserad2026_workdir/runs/se_photonmri_f0/state.pt}
CLF=${2:-/data/kwang/sct_classify_runs/clf_whole/best.pt}
# model config must match the weights (all-75 finals use a different config than 5CV)
CFG=${3:-configs/experiments/all75/all75_p4_mmB.yaml}
DST=container/photon_mri/weights
mkdir -p "$DST"
cp "$W" "$DST/photon_mri.pt"
cp "$CLF" "$DST/clf_whole.pt"
cp /data/kwang/DoseRad2026_raw/beam_parameters.json "$DST/beam_parameters.json"
# deploy config: only model-arch keys are read by E2E/app (training paths are ignored)
cp "$CFG" "$DST/model_config.yaml"
echo "[build] config: $CFG"
echo "[build] staged $(du -h $DST/photon_mri.pt | cut -f1) photon_mri.pt + $(du -h $DST/clf_whole.pt | cut -f1) clf"
docker build -f container/photon_mri/Dockerfile -t doserad-photon-mri:latest .
echo "[build] done -> doserad-photon-mri:latest"
