"""One pass over the raw photon dose to produce BOTH targets we need.

Two separate builders (`build_gt_plans.py`, `build_photon_dose_cache.py`) each traverse the same
383 GB of compressed per-CP .mha. On this box that traversal is the whole cost -- the files are
compressed and there are only 4 cores, so decompression, not the GPU, is the bottleneck (measured:
~42 MB/s per worker against 81 MB/s single-reader / 102 MB/s two-reader disk throughput). Reading
each file once and emitting both outputs halves it.

Per patient, per control point, from a single read:
  * accumulate the TRUE full-grid GT plan          -> gt_plans/<pid>.npy
  * crop to the margin-N aperture bbox             -> photon_dose_m<N>/<pid>/<b>_<cp>.npz

Bboxes come from `photon_channels_fast` (the container's own builder) so the cached target lines up
exactly with what training and inference will see; only the target is stored, since the channels
are 83% of the old cache's bytes and cost 8.2 ms/CP to rebuild.

Shard across processes to use both GPUs and ~2 disk readers (more does not help -- disk-limited):
  CUDA_VISIBLE_DEVICES=0 python scripts/build_photon_targets.py --margin 24 --shard 0/2 &
  CUDA_VISIBLE_DEVICES=1 python scripts/build_photon_targets.py --margin 24 --shard 1/2 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--margin", type=int, default=24)
ap.add_argument("--dose-out", default=None)
ap.add_argument("--gt-out", default=None)
ap.add_argument("--shard", default="0/1", help="i/n — process patients where index %% n == i")
ap.add_argument("--patients", default=None)
a = ap.parse_args()
os.environ["DOSERAD_PHOTON_MARGIN"] = str(a.margin)

from accel.photon_channels_fast import photon_channels_fast   # noqa: E402
from doserad.physics.density import hu_to_density             # noqa: E402
from doserad.physics.machine import load_photon_machine       # noqa: E402
from doserad.inference.pipeline import _build_coords          # noqa: E402
from container.photon.app import _Img                         # noqa: E402

ROOT = Path("/data/kwang/DoseRad2026_raw/photon/training")
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
DOSE_OUT = Path(a.dose_out) if a.dose_out else Path("/data/kwang/doserad_cache") / f"photon_dose_m{a.margin}"
GT_OUT = Path(a.gt_out) if a.gt_out else WORK / "cache" / "gt_plans" / "photon"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

i_shard, n_shard = (int(x) for x in a.shard.split("/"))
machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
pids = a.patients.split(",") if a.patients else \
    sorted(p.name for p in ROOT.iterdir() if (p / "dose").is_dir())
mine = [p for k, p in enumerate(pids) if k % n_shard == i_shard]
GT_OUT.mkdir(parents=True, exist_ok=True)
print(f"shard {i_shard}/{n_shard}: {len(mine)} patients, margin {a.margin}, dev {DEV}", flush=True)

for n_done, pid in enumerate(mine, 1):
    plan = json.load(open(ROOT / pid / f"{pid}.json"))
    n_cp = sum(len(b["control_points"]) for b in plan["beams"])
    dst = DOSE_OUT / pid
    gtf = GT_OUT / f"{pid}.npy"
    if gtf.exists() and dst.exists() and len(list(dst.glob("*.npz"))) == n_cp:
        print(f"[{n_done}/{len(mine)}] {pid}: complete, skip", flush=True)
        continue
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # bboxes first: one cheap GPU pass, so the file read below happens exactly once
    ct_sitk = sitk.ReadImage(str(ROOT / pid / "image" / "ct.mha"))
    img = _Img(ct_sitk)
    dens_t = torch.as_tensor(hu_to_density(img.array, machine.hu_anchors), device=DEV)
    coords = _build_coords(img, DEV)
    bboxes = {}
    with torch.no_grad():
        for bi, b in enumerate(plan["beams"]):
            iso = np.asarray(b.get("iso_center", [0, 0, 0]), np.float64)
            for ci, cp in enumerate(b["control_points"]):
                _, bb = photon_channels_fast(
                    img, machine, iso, cp["gantry_angle"],
                    np.asarray(cp["mlc_left_int_mm"]), np.asarray(cp["mlc_right_int_mm"]),
                    dens_t=dens_t, coords=coords, crop_margin=a.margin, dev=DEV)
                bboxes[(bi, ci)] = tuple(int(v) for v in bb)
    del dens_t, coords
    torch.cuda.empty_cache()
    t_bbox = time.time() - t0

    acc = np.zeros(img.array.shape, np.float32)
    mb = 0.0
    for (bi, ci), bb in bboxes.items():
        f = dst / f"{bi}_{ci:03d}.npz"
        dose = sitk.GetArrayFromImage(
            sitk.ReadImage(str(ROOT / pid / "dose" / f"Dose_B{bi}_CP{ci:03d}.mha")))
        acc += dose                                    # the true full-grid plan
        if not f.exists():
            z0, z1, y0, y1, x0, x1 = bb
            crop = dose[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].astype(np.float16)
            # COMPRESSED, like the existing cache: dose is mostly zeros, so this is ~3.5x smaller
            # (5.7 GB -> ~1.6 GB per patient). That matters more than the decompression cost --
            # uncompressed the cohort is ~425 GB against 125 GB of RAM, so training would miss page
            # cache and read 42 MB/step from an 81 MB/s disk (~0.5 s/step). Compressed it mostly
            # stays resident. The old pipeline trained off a compressed cache, so this is proven.
            np.savez_compressed(f, dose=crop, bbox=np.asarray(bb, np.int32),
                                dose_max=np.float32(dose.max()))
            mb += crop.nbytes / 2**20
    if not gtf.exists():
        tmp = gtf.with_suffix(".tmp.npy")
        np.save(tmp, acc)
        tmp.replace(gtf)
    print(f"[{n_done}/{len(mine)}] {pid}: {len(bboxes)} CPs, bbox {t_bbox:.0f}s, "
          f"total {time.time()-t0:.0f}s, dose {mb:.0f} MB, plan max {acc.max():.3e}", flush=True)
