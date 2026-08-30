"""Does the GPU gamma agree with pymedphys? Do not trust it until this passes.

A fast metric that disagrees with the scorer is worse than a slow one -- this whole phase exists
because we spent weeks optimising against a metric that did not match the platform. So compare
per-voxel, not just the pass rate: two implementations can land on the same percentage while
disagreeing on which voxels fail.

  python scripts/validate_gamma_gpu.py [--patients PID,PID] [--margin 24]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--patients", default="1ABB006")
ap.add_argument("--margin", type=int, default=24)
ap.add_argument("--ckpt", default="/home/kaiwang/doserad2026_workdir/runs/all75_extracted/p4_dosenet.pt")
a = ap.parse_args()
os.environ["DOSERAD_PHOTON_MARGIN"] = str(a.margin)
os.environ.setdefault("DOSERAD_WEIGHTS", a.ckpt)
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")

import torch                                                     # noqa: E402
from doserad.physics.density import hu_to_density                # noqa: E402
from doserad.physics.machine import load_photon_machine          # noqa: E402
from container.photon import app                                 # noqa: E402
from container.photon.predict import predict_cps                 # noqa: E402
from container.proton.gc_invoke import _apply_cutoff             # noqa: E402
from doserad.eval.gamma_gpu import gamma_array_gpu               # noqa: E402

ROOT = Path("/data/kwang/DoseRad2026_raw/photon/training")
GT = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir")) / "cache" / "gt_plans" / "photon"
CUT = 3.328e-7

app.load_models()
machine = load_photon_machine(os.environ["DOSERAD_MACHINE"])

for pid in a.patients.split(","):
    gtf = GT / f"{pid}.npy"
    ct = sitk.ReadImage(str(ROOT / pid / "image" / "ct.mha"))
    img = app._Img(ct)
    if gtf.exists():
        gt = np.load(gtf).astype(np.float32)
    else:
        gt = np.zeros(img.array.shape, np.float32)
        for f in sorted(glob.glob(str(ROOT / pid / "dose" / "*.mha"))):
            gt += sitk.GetArrayFromImage(sitk.ReadImage(f))

    plan = json.load(open(ROOT / pid / f"{pid}.json"))
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        beams.append(b)
    dens = hu_to_density(img.array, machine.hu_anchors)
    preds = predict_cps(img, beams, dens, app._STATE["net"], machine, "cuda")
    pp = np.zeros(img.array.shape, np.float32)
    for crop, bbox in preds.values():
        c = _apply_cutoff(crop, {"minimum_cutoff": CUT})
        z0, z1, y0, y1, x0, x1 = bbox
        pp[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += c

    sp = ct.GetSpacing()
    rx = float(gt.max())

    t0 = time.time()
    g_gpu, m_gpu = gamma_array_gpu(pp, gt, sp, rx, 1.0, 1.0)
    torch.cuda.synchronize()
    t_gpu = time.time() - t0

    from doserad.eval.gamma import gamma_array                    # imported late: see cv_eval note
    t0 = time.time()
    g_cpu, m_cpu = gamma_array(pp, gt, sp, rx, dose_pct=1.0, dta_mm=1.0)
    t_cpu = time.time() - t0

    assert np.array_equal(m_gpu, m_cpu), "evaluation masks differ"
    a_gpu, a_cpu = g_gpu[m_gpu], g_cpu[m_cpu]
    finite = np.isfinite(a_cpu) & np.isfinite(a_gpu)
    p_gpu = float((a_gpu <= 1).mean()) * 100
    p_cpu = float((a_cpu <= 1).mean()) * 100
    disagree = float(((a_gpu <= 1) != (a_cpu <= 1)).mean()) * 100

    print(f"\n=== {pid} ===  {m_gpu.sum():,} evaluated voxels")
    print(f"  pymedphys  {t_cpu:7.1f}s   pass {p_cpu:6.3f}%")
    print(f"  GPU        {t_gpu:7.1f}s   pass {p_gpu:6.3f}%   ({t_cpu/max(t_gpu,1e-9):.1f}x faster)")
    print(f"  pass-rate delta {p_gpu - p_cpu:+.4f} pp | per-voxel verdict disagreement {disagree:.4f}%")
    print(f"  gamma value: mean|d| {np.abs(a_gpu[finite]-a_cpu[finite]).mean():.5f}  "
          f"max|d| {np.abs(a_gpu[finite]-a_cpu[finite]).max():.5f}")
