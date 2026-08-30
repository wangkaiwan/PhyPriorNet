"""Photon-CT container gamma test — WHICH weights should the photon-CT image carry?

The platform scored our photon-CT at gamma 1%/1mm 78.46 while the internal 5CV number is 95.7
(docs/internal/STATUS.md), a 17-point gap that generalisation does not explain (proton-CT dropped
only 97.0 -> 95.05 on the same platform). The shipped weights are the dose sub-net extracted from
all75_p4_mmB, a MULTIMODAL run, so the suspicion is a channel-convention mismatch between how that
sub-net was trained and how container/photon/predict.py builds channels. This runs the exact deploy
path on one patient and scores plan gamma against the MC ground truth, so candidate weights can be
compared directly instead of argued about.

  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python container/photon/test_gamma.py <PID> <weights>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
if len(sys.argv) > 2:
    os.environ["DOSERAD_WEIGHTS"] = sys.argv[2]
os.environ.setdefault("DOSERAD_WEIGHTS",
                      "/home/kaiwang/doserad2026_workdir/runs/all75_extracted/p4_dosenet.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
CACHE = os.environ.get("DOSERAD_CACHE",
                       yaml.safe_load(open("configs/experiments/all75/all75_p1_photonct.yaml"))["cache_dir"])

from container.photon import app                                    # noqa: E402
from doserad.eval.plan_agg import accumulate_plan                   # noqa: E402
from doserad.eval.gamma import gamma_array                          # noqa: E402

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"


def main():
    print(f"weights: {os.environ['DOSERAD_WEIGHTS']}")
    print(f"cache  : {CACHE}")
    app.load_models()
    ct = sitk.ReadImage(f"{ROOT}/{PID}/image/ct.mha")
    plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for ci, cp in enumerate(b["control_points"]):
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    entry = {"image_file_idx": 0, "beams": beams}

    t0 = time.time()
    preds = app._predict_fn(ct, entry)
    dt = time.time() - t0

    full = sitk.GetArrayFromImage(ct).shape
    pred_cps, gt_cps = [], []
    for f in sorted((Path(CACHE) / PID).glob("*.npz")):
        bi, cpi = (int(x) for x in f.stem.split("_"))
        z = np.load(f)
        gt_cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
        if (bi, cpi) in preds:
            crop, pbb, _ = preds[(bi, cpi)]
            pred_cps.append((crop, pbb))
    pp = accumulate_plan(pred_cps, full)
    gt = accumulate_plan(gt_cps, full)
    rx = float(gt.max())
    zz, yy, xx = np.where(gt >= 0.05 * rx)
    m = 4
    crop = (slice(max(int(zz.min()) - m, 0), int(zz.max()) + m + 1),
            slice(max(int(yy.min()) - m, 0), int(yy.max()) + m + 1),
            slice(max(int(xx.min()) - m, 0), int(xx.max()) + m + 1))
    sp = ct.GetSpacing()
    g1c, g1m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m] <= 1.0).mean()) * 100 if g1m.any() else float("nan")
    g3c, g3m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=3.0, dta_mm=3.0)
    g3 = float((g3c[g3m] <= 1.0).mean()) * 100 if g3m.any() else float("nan")
    print(f"\n=== PHOTON-CT CONTAINER GAMMA ({PID}) ===")
    print(f"  {len(pred_cps)}/{len(gt_cps)} CPs predicted, {dt:.1f}s "
          f"({dt/max(len(pred_cps),1)*1000:.0f} ms/CP)")
    print(f"  plan gamma 1%/1mm {g1:.1f}%   3%/3mm {g3:.1f}%   "
          f"(internal 5CV photon-CT 95.7; platform scored 78.46)")


if __name__ == "__main__":
    main()
