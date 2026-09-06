#!/bin/bash
# Stage photon weights (NOT in git) + build the photon GC invoke image. Run from repo root:
#   bash container/photon/build.sh [WEIGHTS_PT]
set -eu
cd "$(git rev-parse --show-toplevel)"
W=${1:-/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt}
DST=container/photon/weights
mkdir -p "$DST"
cp "$W" "$DST/photon.pt"
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
echo "[build] staged $(du -h $DST/photon.pt | cut -f1) photon.pt"
docker build -f container/photon/Dockerfile -t doserad-photon:latest .
echo "[build] done -> doserad-photon:latest"
