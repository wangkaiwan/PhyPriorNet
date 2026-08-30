#!/bin/bash
# Zero-cutoff PROBE for photon-MRI: layer the toggle-carrying gc_invoke.py + ENV DOSERAD_CUTOFF_ZERO=1
# onto the shipped shift-robust champion (m24S2ft_mraug 332c7d29 + clf_whole_mraug 7892ba1f, margin-24).
# ONLY the cutoff application changes (zero `dose<=cutoff=0` vs default quantise). Clean minimal pair.
set -eu
cd "$(git rev-parse --show-toplevel)"
SUB=/data/kwang/doserad_submissions
BASE=doserad-photon-mri:ft20k-shiftrobust
OUT=doserad-photon-mri_zerocut_m24.tar.gz
CTX=$(mktemp -d); cp container/proton/gc_invoke.py "$CTX/gc_invoke.py"
grep -q "DOSERAD_CUTOFF_ZERO" "$CTX/gc_invoke.py" || { echo "FATAL: no toggle"; exit 1; }
printf 'FROM %s\nCOPY gc_invoke.py /opt/algorithm/container/proton/gc_invoke.py\nENV DOSERAD_CUTOFF_ZERO=1\n' "$BASE" > "$CTX/Dockerfile"
echo "[build] doserad-photon-mri:zerocut-m24 FROM $BASE"
sg docker -c "docker build -q -t doserad-photon-mri:zerocut-m24 $CTX"
echo "[verify] baked env:"
sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' doserad-photon-mri:zerocut-m24" | grep -iE "CUTOFF_ZERO|MARGIN|COMPRESS|WEIGHTS|CLF"
echo "[save] -> $OUT"
sg docker -c "docker save doserad-photon-mri:zerocut-m24" | gzip -1 > "$SUB/$OUT"
echo "[done] $(du -h "$SUB/$OUT" | cut -f1)  $OUT"
md5sum "$SUB/$OUT" | sed 's#/data.*/##'
rm -rf "$CTX"
echo "PHOTON-MRI ZERO-PROBE BUILT"
