#!/bin/bash
# Stage weights (NOT in git) + build the proton GC invoke image. Run from repo root:
#   bash container/proton/build.sh [WEIGHTS_PT]
set -eu
cd "$(git rev-parse --show-toplevel)"
W=${1:-/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt}
DST=container/proton/weights
mkdir -p "$DST"
cp "$W" "$DST/proton.pt"
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
# proton machine kernel npz (ProtonMachineData default) — bake it too if not already in doserad/
echo "[build] staged weights: $(du -h $DST/proton.pt | cut -f1) proton.pt"
docker build -f container/proton/Dockerfile -t doserad-proton:latest .
echo "[build] done -> doserad-proton:latest"
