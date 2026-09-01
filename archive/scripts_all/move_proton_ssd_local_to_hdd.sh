#!/bin/bash
# Move the UNUSED proton_ssd_local crop cache (200G, last touched 2026-06-21, only old CV fold-0 configs
# reference it; Step 4 uses proton_ssd symlink not this) off the root SSD to the HDD, leaving a symlink
# so any path reference still resolves. rm -rf is permission-blocked, so delete source via find -delete.
set -eu
SRC=/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd_local
DST=/data/kwang/doserad_cache_hdd/proton_ssd_local
echo "[move] rsync $SRC -> $DST"
mkdir -p "$DST"
rsync -a "$SRC/" "$DST/"
NS=$(find "$SRC" -type f | wc -l); ND=$(find "$DST" -type f | wc -l)
echo "[move] file counts: src=$NS dst=$ND"
if [ "$NS" -ne "$ND" ] || [ "$ND" -eq 0 ]; then echo "[move] ABORT: count mismatch, source kept"; exit 1; fi
echo "[move] verified -> deleting source + symlinking"
find "$SRC" -type f -delete
find "$SRC" -type d -empty -delete
[ -e "$SRC" ] && rmdir "$SRC" 2>/dev/null || true
ln -s "$DST" "$SRC"
ls -ld "$SRC"
echo "[move] done. SSD now:"; df -h / | tail -1
echo "PROTON_SSD_LOCAL MOVED TO HDD"
