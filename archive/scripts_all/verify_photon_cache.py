"""Scan a photon crop cache for truncated .npz files and remove them so they get rebuilt.

Why this exists: the margin-24 cache build hit ENOSPC at 61/75 patients, and each of the two
parallel shards left behind one half-written .npz. Those files still `exists()`, so the resume
skipped them and the corruption would have reached training silently.

The check is exact for truncation rather than heuristic: an .npz is a zip, and a zip's central
directory sits at the END of the file, so any truncation destroys it and `np.load` raises
BadZipFile. Reading the central directory does not decompress anything, so a full 40k-file cache
scans in well under a minute.

Prints the number removed and exits 0 (nothing wrong) or 1 (files were removed -> rebuild needed).

  python scripts/verify_photon_cache.py /path/to/cache [--jobs 16]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("--jobs", type=int, default=16)
ap.add_argument("--dry-run", action="store_true")
# The dose-only source cache from build_photon_targets.py has no `channels`, so the key set must
# be told, not assumed -- pointing the default at that cache would flag every file as missing a
# key and delete the lot.
ap.add_argument("--keys", default="channels,dose,bbox,dose_max",
                help="comma-separated keys every npz must contain")
a = ap.parse_args()
REQUIRED = {k for k in a.keys.split(",") if k}

cache = Path(a.cache)
files = sorted(cache.glob("*/*.npz"))
print(f"scanning {len(files)} npz under {cache}", flush=True)


def check(f: Path):
    try:
        z = np.load(f)
        missing = REQUIRED - set(z.files)
        return None if not missing else f"missing keys {sorted(missing)}"
    except Exception as e:                     # BadZipFile, EOFError, OSError...
        return f"{type(e).__name__}: {e}"


bad = []
with ThreadPoolExecutor(max_workers=a.jobs) as ex:
    for f, err in zip(files, ex.map(check, files)):
        if err:
            bad.append((f, err))
            print(f"  CORRUPT {f.relative_to(cache)}: {err}", flush=True)

# stale .part files from an interrupted atomic write are harmless (the *.npz completeness glob
# ignores them) but there is no reason to keep them around
parts = list(cache.glob("*/*.part"))

if not a.dry_run:
    for f, _ in bad:
        f.unlink()
    for p in parts:
        p.unlink()

verb = "would remove" if a.dry_run else "removed"
print(f"{verb} {len(bad)} corrupt npz + {len(parts)} stale .part", flush=True)
print(f"affected patients: {sorted({f.parent.name for f, _ in bad})}" if bad else "cache clean",
      flush=True)
sys.exit(1 if bad else 0)
