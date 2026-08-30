#!/bin/bash
# Zero-cutoff PROBE: layer the toggle-carrying gc_invoke.py + ENV DOSERAD_CUTOFF_ZERO=1 onto an
# ALREADY-SHIPPED champion image, changing NOTHING else (same weights/margin/compress that scored on
# the leaderboard). This makes a clean minimal pair vs the shipped baseline: the ONLY variable that
# differs is how the per-beam minimum_cutoff is applied (zero `dose<=cutoff=0` vs the default quantise).
# See email 2026-08-07 change #5 (GT swapped to a cutoff-adhering version).
set -eu
cd "$(git rev-parse --show-toplevel)"
SUB=/data/kwang/doserad_submissions
CTX=$(mktemp -d); cp container/proton/gc_invoke.py "$CTX/gc_invoke.py"
grep -q "DOSERAD_CUTOFF_ZERO" "$CTX/gc_invoke.py" || { echo "FATAL: gc_invoke.py missing the toggle"; exit 1; }
echo "[ok] staged gc_invoke.py carries the CUTOFF_ZERO toggle"

build_zero(){  # $1=base image  $2=out tag  $3=out tar
  printf 'FROM %s\nCOPY gc_invoke.py /opt/algorithm/container/proton/gc_invoke.py\nENV DOSERAD_CUTOFF_ZERO=1\n' "$1" > "$CTX/Dockerfile"
  echo "[build] $2  FROM $1"
  sg docker -c "docker build -q -t $2 $CTX"
  echo "[verify] baked env:"
  sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' $2" | grep -iE "CUTOFF_ZERO|MARGIN|COMPRESS|MODALITY"
  echo "[save] -> $3"
  sg docker -c "docker save $2" | gzip -1 > "$SUB/$3"
  echo "[done] $(du -h "$SUB/$3" | cut -f1)  $3"
}

build_zero doserad-photon:p2                 doserad-photon:zerocut-m24  doserad-photon-ct_zerocut_m24.tar.gz
build_zero doserad-proton:all75r2-compressed doserad-proton:zerocut      doserad-proton-ct_zerocut.tar.gz
rm -rf "$CTX"
echo "ALL ZERO-PROBE DOCKERS BUILT"
