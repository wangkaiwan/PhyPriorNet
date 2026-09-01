"""Photon dose-target cache at an arbitrary aperture margin, WITHOUT the channels.

Two decisions are baked in here, both consequences of what went wrong on 2026-07-24:

1. **Dose only.** In the existing cache the 5 channels are 83% of the bytes (9.59 MB vs 1.92 MB per
   CP) and they are fully reproducible from the CT + plan by `photon_channels_fast`, at 8.2 ms/CP --
   negligible against a training step. Storing only the target makes a margin-24 cache ~94 GB
   instead of ~565 GB, so it fits alongside the existing one.

2. **Channels computed live at train time, from the deploy code path.** The model was trained on
   margin-8 crops and then deployed at margin 24, which is out-of-distribution in crop SIZE
   (padding boundary and receptive-field context both change) and showed up as a small but
   rank-costing beam-MAE regression on the platform. Training through the same builder the
   container uses removes that whole class of skew.

The bbox is taken from `photon_channels_fast` itself rather than recomputed, so the cached target
lines up exactly with what training and inference will see.

  python scripts/build_photon_dose_cache.py --margin 24 --out DIR [--patients PID,PID]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--margin", type=int, default=24)
ap.add_argument("--out", default=None)
ap.add_argument("--patients", default=None)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
os.environ["DOSERAD_PHOTON_MARGIN"] = str(a.margin)      # read at import by container.photon.predict

from accel.photon_channels_fast import photon_channels_fast   # noqa: E402
from doserad.physics.density import hu_to_density             # noqa: E402
from doserad.physics.machine import load_photon_machine       # noqa: E402
from doserad.inference.pipeline import _build_coords          # noqa: E402
from container.photon.app import _Img                         # noqa: E402

ROOT = Path("/data/kwang/DoseRad2026_raw/photon/training")
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
OUT = Path(a.out) if a.out else Path("/data/kwang/doserad_cache") / f"photon_dose_m{a.margin}"
DEV = "cuda"

machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
pids = a.patients.split(",") if a.patients else \
    sorted(p.name for p in ROOT.iterdir() if (p / "dose").is_dir())

for i, pid in enumerate(pids, 1):
    dst = OUT / pid
    plan = json.load(open(ROOT / pid / f"{pid}.json"))
    n_cp = sum(len(b["control_points"]) for b in plan["beams"])
    if dst.exists() and len(list(dst.glob("*.npz"))) == n_cp and not a.force:
        print(f"[{i}/{len(pids)}] {pid}: complete, skip", flush=True)
        continue
    dst.mkdir(parents=True, exist_ok=True)

    ct_sitk = sitk.ReadImage(str(ROOT / pid / "image" / "ct.mha"))
    img = _Img(ct_sitk)
    dens_t = torch.as_tensor(hu_to_density(img.array, machine.hu_anchors), device=DEV)
    coords = _build_coords(img, DEV)

    n, mb = 0, 0.0
    with torch.no_grad():
        for bi, b in enumerate(plan["beams"]):
            iso = np.asarray(b.get("iso_center", [0, 0, 0]), np.float64)
            for ci, cp in enumerate(b["control_points"]):
                f = dst / f"{bi}_{ci:03d}.npz"
                if f.exists() and not a.force:
                    continue
                _, bbox = photon_channels_fast(
                    img, machine, iso, cp["gantry_angle"],
                    np.asarray(cp["mlc_left_int_mm"]), np.asarray(cp["mlc_right_int_mm"]),
                    dens_t=dens_t, coords=coords, crop_margin=a.margin, dev=DEV)
                z0, z1, y0, y1, x0, x1 = (int(v) for v in bbox)
                dose = sitk.GetArrayFromImage(
                    sitk.ReadImage(str(ROOT / pid / "dose" / f"Dose_B{bi}_CP{ci:03d}.mha")))
                crop = dose[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].astype(np.float16)
                np.savez(f, dose=crop, bbox=np.asarray(bbox, np.int32),
                         dose_max=np.float32(dose.max()))
                n += 1
                mb += crop.nbytes / 2**20
    print(f"[{i}/{len(pids)}] {pid}: {n} CPs written, {mb:.0f} MB", flush=True)
    del dens_t, coords
    torch.cuda.empty_cache()
