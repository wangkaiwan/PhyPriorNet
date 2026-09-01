"""Build a SLIM per-CP dose cache on SSD for the future end-to-end MRI dose model.

The end-to-end model recomputes density/rdepth/fluence/naive FROM the predicted sCT (differentiably)
and computes geometry (dist/source) on the fly, so it only needs the GT **dose** (+ bbox) per CP.
This copies just dose + bbox + dose_max out of the big HDD crop cache into a small SSD cache
(~36G vs 214G), removing the per-step HDD read at training time.

Source (HDD): /data/kwang/doserad_cache_archive/photon_crops/<pid>/<B>_<CP>.npz  (channels,dose,bbox,..)
Dest  (SSD): $DOSERAD_WORK/cache/crops/photon_dose_ssd/<pid>/<B>_<CP>.npz        (dose,bbox,dose_max)
np.load reads only the requested members, so the big `channels` array is never read from disk.

Resumable (skips existing). Atomic writes. Aborts a patient if SSD free < SSD_MIN_GB (default 210).
Usage: conda run -n doserad python scripts/build_dose_slim_ssd.py [pid ...]
"""
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
import numpy as np

SRC = Path("/data/kwang/doserad_cache_archive/photon_crops")
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
DST = WORK / "cache" / "crops" / "photon_dose_ssd"
SSD_MIN_GB = float(os.environ.get("SSD_MIN_GB", "210"))   # stay well above the 200G alert


def ssd_free_gb() -> float:
    u = shutil.disk_usage("/")
    return u.free / 1e9


def process_patient(pid: str) -> int:
    sdir = SRC / pid
    if not sdir.is_dir():
        return 0
    odir = DST / pid
    odir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(sdir.glob("*.npz")):
        if ".tmp" in src.name:
            continue
        out = odir / src.name
        if out.exists():
            continue
        z = np.load(src)                      # lazy; we read only dose/bbox/dose_max members
        dose = z["dose"]; bbox = z["bbox"]
        dmax = z["dose_max"] if "dose_max" in z.files else np.float32(float(dose.max()))
        tmp = out.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, dose=dose.astype(np.float16), bbox=bbox.astype(np.int32),
                            dose_max=np.float32(dmax))
        os.replace(tmp, out)
        n += 1
    return n


def main(pids):
    if not pids:
        pids = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    DST.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        free = ssd_free_gb()
        if free < SSD_MIN_GB:
            print(f"[ABORT] SSD free {free:.0f}G < {SSD_MIN_GB:.0f}G — stopping before {pid}", flush=True)
            break
        n = process_patient(pid)
        print(f"{pid}: wrote {n} dose crops  (SSD free {ssd_free_gb():.0f}G)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
