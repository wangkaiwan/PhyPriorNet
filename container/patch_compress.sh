#!/bin/bash
# Layer the compressed-output gc_invoke.py onto an ALREADY-SHIPPED container image, changing NOTHING
# else — same weights, same config, same everything that scored on the leaderboard. This is the safe
# way to ship the compression win without rebuilding from weights (build.sh defaults to a 5CV fold-0
# checkpoint, so a from-scratch rebuild risks silently swapping the model — see the submission memo).
#
# gc_invoke.py is shared by all four containers and lives at the same path inside every image
# (/opt/algorithm/container/proton/gc_invoke.py), so one COPY patches any of them.
#
#   bash container/patch_compress.sh <task>
#     task = proton-ct | photon-ct | photon-mri | proton-mri
#
# It loads the newest shipped tar for that task, builds <image>:compress FROM it with the current
# gc_invoke.py, and stops there. It does NOT export (docker save) and does NOT push — do that
# yourself when you actually submit. Verify with the smoke harness before trusting it.
set -eu
cd "$(git rev-parse --show-toplevel)"
TASK=${1:?usage: patch_compress.sh <proton-ct|photon-ct|photon-mri|proton-mri>}
SUB=/data/kwang/doserad_submissions

# newest shipped tar + the repo:tag it loads to, per task
case "$TASK" in
  proton-ct)  TAR=$SUB/doserad-proton-ct_v9_nocompile.tar.gz;              TAG=doserad-proton:latest;;
  photon-ct)  TAR=$SUB/doserad-photon-ct_v2_cutoff+margin24.tar.gz;        TAG=doserad-photon:latest;;
  photon-mri) TAR=$SUB/doserad-photon-mri_v4_cutoff+margin24-REAL.tar.gz;  TAG=doserad-photon-mri:latest;;
  proton-mri) TAR=$SUB/doserad-proton-mri_v4_nocompile.tar.gz;             TAG=doserad-proton-mri:latest;;
  *) echo "unknown task: $TASK"; exit 1;;
esac
[ -f "$TAR" ] || { echo "shipped tar not found: $TAR"; exit 1; }

echo "[patch] loading shipped image: $(basename "$TAR")"
LOADED=$(docker load -i "$TAR" | sed -n 's/^Loaded image: //p' | head -1)
BASE=${LOADED:-$TAG}
echo "[patch] base image = $BASE"

# minimal build context: just the new gc_invoke.py
CTX=$(mktemp -d)
cp container/proton/gc_invoke.py "$CTX/gc_invoke.py"
cat > "$CTX/Dockerfile" <<EOF
FROM $BASE
COPY gc_invoke.py /opt/algorithm/container/proton/gc_invoke.py
EOF

OUT="doserad-${TASK}:compress"
echo "[patch] building $OUT (weights untouched, only gc_invoke.py replaced)"
docker build -f "$CTX/Dockerfile" -t "$OUT" "$CTX"
rm -rf "$CTX"

# prove ONLY gc_invoke.py changed: the weight md5 must match the shipped image
echo "[patch] weight md5 (must be identical shipped vs patched):"
for img in "$BASE" "$OUT"; do
  docker run --rm --entrypoint sh "$img" -c \
    'md5sum /opt/algorithm/container/*/weights/*.pt 2>/dev/null' | sed "s/^/    $img  /"
done
echo "[patch] done -> $OUT   (NOT exported; run smoke, then docker save when you submit)"
