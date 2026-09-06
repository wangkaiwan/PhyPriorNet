#!/bin/bash
# Stage proton-MRI weights (E2E synth+dose + sCT classifier, NOT in git) + deploy config + machine,
# then build the GC invoke image. Run from repo root:
#   bash container/proton_mri/build.sh [E2E_STATE_PT] [CLF_PT]
set -eu
cd "$(git rev-parse --show-toplevel)"
W=${1:-/home/kaiwang/doserad2026_workdir/runs/se_protonmri_f0/state.pt}
CLF=${2:-/data/kwang/sct_classify_runs/clf_whole/best.pt}
# model config must match the weights (all-75 finals use a different config than 5CV)
CFG=${3:-configs/experiments/all75/all75_r3_protonmri.yaml}
TAG=${4:-doserad-proton-mri:latest}
DST=container/proton_mri/weights
mkdir -p "$DST"
cp "$W" "$DST/proton_mri.pt"
cp "$CLF" "$DST/clf_whole.pt"
# beam_parameters.json: the released weight bundles already contain it, so an outside user who
# untarred into weights/ is complete. Refresh from the local raw copy only where one exists (our
# dev machines); fail loudly only if NEITHER source is available.
BP_SRC=/data/kwang/DoseRad2026_raw/beam_parameters.json
if [ -f "$BP_SRC" ]; then
  cp "$BP_SRC" "$DST/beam_parameters.json"
elif [ ! -f "$DST/beam_parameters.json" ]; then
  echo "ERROR: $DST/beam_parameters.json missing and no local source at $BP_SRC." >&2
  echo "       It ships inside every release bundle -- untar the bundle into $DST first." >&2
  exit 1
fi
cp "$CFG" "$DST/model_config.yaml"
echo "[build] config: $CFG"
echo "[build] staged $(du -h $DST/proton_mri.pt | cut -f1) proton_mri.pt + $(du -h $DST/clf_whole.pt | cut -f1) clf"
docker build -f container/proton_mri/Dockerfile -t "$TAG" .
echo "[build] done -> $TAG"
