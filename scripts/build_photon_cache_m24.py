"""Complete the margin-24 photon cache: add the 5 physics channels next to the dose already cached.

`build_photon_targets.py` stored only the dose target, on the assumption that channels would be
computed live at train time. Training on all 75 at margin 24 is the decision now, and changing the
data pipeline to compute channels inside DataLoader workers (which cannot hold a CUDA context) is a
real refactor -- exactly the kind of rushed change that has cost us twice already. Disk is not the
constraint on /data (15 TB free), so materialise the channels instead and leave the pipeline alone.

This does NOT re-read the raw per-CP dose (383 GB): the bbox is taken from the cached npz and the
dose crop is copied straight out of it, so the only new work is the GPU channel build (~8 ms/CP).
Output matches what PhotonCropDataset expects: channels (5, z, y, x) float16, dose, bbox, dose_max.

  CUDA_VISIBLE_DEVICES=0 python scripts/build_photon_cache_m24.py --shard 0/2 &
  CUDA_VISIBLE_DEVICES=1 python scripts/build_photon_cache_m24.py --shard 1/2 &
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
ap.add_argument("--src", default=None, help="dose-only cache from build_photon_targets.py")
ap.add_argument("--out", default=None)
ap.add_argument("--shard", default="0/1")
a = ap.parse_args()
os.environ["DOSERAD_PHOTON_MARGIN"] = str(a.margin)

from accel.photon_channels_fast import photon_channels_fast   # noqa: E402
from doserad.physics.density import hu_to_density             # noqa: E402
from doserad.physics.machine import load_photon_machine       # noqa: E402
from doserad.inference.pipeline import _build_coords          # noqa: E402
from container.photon.app import _Img                         # noqa: E402

ROOT = Path("/data/kwang/DoseRad2026_raw/photon/training")
SRC = Path(a.src) if a.src else Path("/data/kwang/doserad_cache") / f"photon_dose_m{a.margin}"
OUT = Path(a.out) if a.out else Path("/data/kwang/doserad_cache") / f"photon_skinentry_m{a.margin}"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

i_shard, n_shard = (int(x) for x in a.shard.split("/"))
machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
pids = sorted(p.name for p in SRC.iterdir() if p.is_dir())
mine = [p for k, p in enumerate(pids) if k % n_shard == i_shard]
print(f"shard {i_shard}/{n_shard}: {len(mine)} patients -> {OUT}", flush=True)

for n, pid in enumerate(mine, 1):
    src_d, dst_d = SRC / pid, OUT / pid
    plan = json.load(open(ROOT / pid / f"{pid}.json"))
    n_cp = sum(len(b["control_points"]) for b in plan["beams"])
    if dst_d.exists() and len(list(dst_d.glob("*.npz"))) == n_cp:
        print(f"[{n}/{len(mine)}] {pid}: complete, skip", flush=True)
        continue
    dst_d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    img = _Img(sitk.ReadImage(str(ROOT / pid / "image" / "ct.mha")))
    dens_t = torch.as_tensor(hu_to_density(img.array, machine.hu_anchors), device=DEV)
    coords = _build_coords(img, DEV)
    written = 0
    gb = 0.0
    with torch.no_grad():
        for bi, b in enumerate(plan["beams"]):
            iso = np.asarray(b.get("iso_center", [0, 0, 0]), np.float64)
            for ci, cp in enumerate(b["control_points"]):
                dst = dst_d / f"{bi}_{ci:03d}.npz"
                if dst.exists():
                    continue
                z = np.load(src_d / f"{bi}_{ci:03d}.npz")
                geom, bb = photon_channels_fast(
                    img, machine, iso, cp["gantry_angle"],
                    np.asarray(cp["mlc_left_int_mm"]), np.asarray(cp["mlc_right_int_mm"]),
                    dens_t=dens_t, coords=coords, crop_margin=a.margin, dev=DEV)
                assert tuple(int(v) for v in bb) == tuple(int(v) for v in z["bbox"]), \
                    f"bbox drift at {pid} {bi}_{ci}"
                ch = geom.detach().to(torch.float16).cpu().numpy()
                # Write-then-rename: a crash mid-write (we hit ENOSPC once, at 61/75) otherwise
                # leaves a truncated .npz that still `exists()`, so the resume skips it and the
                # corruption reaches training. `.part` does not match the *.npz completeness glob.
                tmp = dst.with_name(dst.name + ".part")
                with open(tmp, "wb") as fh:
                    np.savez_compressed(fh, channels=ch, dose=z["dose"],
                                        bbox=z["bbox"], dose_max=z["dose_max"])
                os.replace(tmp, dst)
                written += 1
                gb += (ch.nbytes + z["dose"].nbytes) / 2**30
    del dens_t, coords
    torch.cuda.empty_cache()
    print(f"[{n}/{len(mine)}] {pid}: {written} CPs, {gb:.2f} GB raw, {time.time()-t0:.0f}s", flush=True)
