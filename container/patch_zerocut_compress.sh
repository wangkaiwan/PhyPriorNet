#!/bin/bash
# Add compression to the zero-cutoff photon probes. Zeroed output is >99% zeros so zlib compresses ~100x,
# which cuts the write-bound runtime (photon-CT leaderboard: runtime rank 18 was the only weak spot; both
# runtime columns are double-weighted in Mean Position, so this is the cheapest path to 1st). ONLY the
# DOSERAD_COMPRESS_OUTPUT env flips 0->1; gc_invoke already reads it. Weights/margin/cutoff unchanged.
set -eu
cd "$(git rev-parse --show-toplevel)"
SUB=/data/kwang/doserad_submissions
CTX=$(mktemp -d)

build(){  # $1=base image  $2=out tag  $3=out tar
  printf 'FROM %s\nENV DOSERAD_COMPRESS_OUTPUT=1\n' "$1" > "$CTX/Dockerfile"
  echo "[build] $2  FROM $1"
  sg docker -c "docker build -q -t $2 $CTX"
  echo "[verify] baked env:"
  sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' $2" | grep -iE "CUTOFF_ZERO|COMPRESS|MARGIN|MODALITY"
  echo "[save] -> $3"
  sg docker -c "docker save $2" | gzip -1 > "$SUB/$3"
  echo "[done] $(du -h "$SUB/$3" | cut -f1)  $3"
  md5sum "$SUB/$3" | sed 's#/data.*/##'
}

build doserad-photon:zerocut-m24     doserad-photon:zerocut-m24-compress     doserad-photon-ct_zerocut_m24_compress.tar.gz
build doserad-photon-mri:zerocut-m24 doserad-photon-mri:zerocut-m24-compress doserad-photon-mri_zerocut_m24_compress.tar.gz
rm -rf "$CTX"
echo "PHOTON ZEROCUT+COMPRESS BUILT"
