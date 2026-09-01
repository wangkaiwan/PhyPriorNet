"""Photon-CT plan eval scored the way the CHALLENGE scores it.

`cv_eval_photonct.py` builds the reference plan by summing the CACHED per-CP crops -- the same
aperture-bbox+margin windows the prediction uses -- so prediction and reference are truncated
identically and the metric cannot see what the crop drops. Every internal photon number we have
ever quoted came from that harness. On 1ABB006 the same prediction scores 96.18 against the cropped
reference and 86.38 against the raw full-grid one, and the distilled base32 student -- declared "at
parity with the teacher" under the old harness -- loses 17 points on a hard patient under this one.

Differences from the old script, all in the direction of matching the platform:
  * reference plan = the TRUE full-grid GT (scripts/build_gt_plans.py), not a sum of crops;
  * channels are computed live by the deploy path (container.photon.predict), not read from the
    channels cache, so training/eval/deploy cannot drift apart -- and the crop margin becomes a
    knob (DOSERAD_PHOTON_MARGIN) instead of being frozen at whatever the cache was built with;
  * the platform's minimum_cutoff quantisation is applied to the prediction.

Per-CP metrics are deliberately left alone: 0.00% of a CP's >=10%-of-max voxels fall outside its
crop, so beam MAE and IDD were never biased by it.

  python scripts/cv_eval_photonct_full.py --config CFG --ckpt CKPT --out OUT.csv \
      [--margin 24] [--cutoff 3.328e-7] [--also-cropped]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=int, default=24, help="aperture crop margin in voxels (2 mm each)")
ap.add_argument("--cutoff", type=float, default=3.328e-7,
                help="platform minimum_cutoff; 0 disables the quantiser")
ap.add_argument("--gt-plans", default=None)
ap.add_argument("--also-cropped", action="store_true",
                help="additionally score against the old cropped reference, to size the bias")
ap.add_argument("--patients", default=None)
ap.add_argument("--gpu-gamma", type=int, default=0,
                help="interp_fraction for the GPU gamma; 0 = pymedphys. 10 screens (26x, small "
                     "systematic bias that cancels between checkpoints), 20 reports (3.4x).")
a = ap.parse_args()

# must be set before container.photon.predict is imported (it reads the env at module load)
os.environ["DOSERAD_PHOTON_MARGIN"] = str(a.margin)

from doserad.model.unet3d import DoseUNet3D                      # noqa: E402
from doserad.eval.plan_agg import accumulate_plan, stratified_mae  # noqa: E402
from doserad.eval.gamma import gamma_array, gamma_pass           # noqa: E402
from doserad.physics.density import hu_to_density                # noqa: E402
from doserad.physics.machine import load_photon_machine          # noqa: E402
from doserad.io.mha import load_mha                              # noqa: E402
from container.photon.predict import predict_cps                 # noqa: E402
from container.proton.gc_invoke import _apply_cutoff             # noqa: E402

ROOT = Path("/data/kwang/DoseRad2026_raw/photon/training")
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
GT_PLANS = Path(a.gt_plans) if a.gt_plans else WORK / "cache" / "gt_plans" / "photon"

cfg = yaml.safe_load(open(a.config))
dev = "cuda"
val = a.patients.split(",") if a.patients else \
    json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]

net = DoseUNet3D(in_ch=cfg.get("in_ch", 6), base=cfg["base_ch"], levels=cfg["levels"],
                 bottleneck=cfg.get("bottleneck", "plain")).to(dev).eval()
sd = torch.load(a.ckpt, map_location=dev)
sd = sd.get("ema", sd.get("model", sd))
if any(k.startswith("dose.") for k in sd):        # an E2E checkpoint (synth.* + dose.*)
    sd = {k[len("dose."):]: v for k, v in sd.items() if k.startswith("dose.")}
net.load_state_dict(sd)
machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
cache = Path(cfg["cache_dir"])

rows = []
for pid in val:
    gtf = GT_PLANS / f"{pid}.npy"
    if not gtf.exists():
        print(f"  {pid}: no full-grid GT plan yet ({gtf}) — run scripts/build_gt_plans.py", flush=True)
        continue
    ct = load_mha(ROOT / pid / "image" / "ct.mha")
    full = ct.array.shape
    plan = json.load(open(ROOT / pid / f"{pid}.json"))
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        beams.append(b)

    dens = hu_to_density(ct.array, machine.hu_anchors)
    preds = predict_cps(ct, beams, dens, net, machine, dev)

    pp = np.zeros(full, np.float32)
    for crop, bbox in preds.values():
        c = _apply_cutoff(crop, {"minimum_cutoff": a.cutoff}) if a.cutoff > 0 else crop
        z0, z1, y0, y1, x0, x1 = bbox
        pp[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += c

    gt = np.load(gtf).astype(np.float32)
    rx = float(gt.max())
    if a.gpu_gamma:
        from doserad.eval.gamma_gpu import gamma_array_gpu
        g1c, g1m = gamma_array_gpu(pp, gt, ct.spacing, rx, 1.0, 1.0,
                                   interp_fraction=a.gpu_gamma)
        g3c, g3m = gamma_array_gpu(pp, gt, ct.spacing, rx, 3.0, 3.0,
                                   interp_fraction=a.gpu_gamma)
        g3 = float((g3c[g3m] <= 1).mean()) if g3m.any() else float("nan")
    else:
        g1c, g1m = gamma_array(pp, gt, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        g3 = gamma_pass(pp, gt, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    g1 = float((g1c[g1m] <= 1).mean()) if g1m.any() else float("nan")

    row = {"patient": pid, "site": "lung" if "THB" in pid else "abdomen",
           "plan_g1": g1, "plan_g3": g3, "strat_mae": stratified_mae(pp, gt, rx),
           "pred_energy_frac": float(pp.sum() / gt.sum())}

    if a.also_cropped:                      # the OLD reference, to size the bias it hid
        gc = []
        for bi, b in enumerate(beams):
            for ci, _ in enumerate(b["control_points"]):
                f = cache / pid / f"{bi}_{ci:03d}.npz"
                if f.exists():
                    z = np.load(f)
                    gc.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
        if gc:
            gtc = accumulate_plan(gc, full)
            rxc = float(gtc.max())
            row["plan_g1_croppedref"] = gamma_pass(pp, gtc, ct.spacing, rxc, 1.0, 1.0)
            row["bias"] = row["plan_g1_croppedref"] - g1

    rows.append(row)
    extra = (f"  | cropped-ref {row['plan_g1_croppedref']*100:.1f} "
             f"(bias +{row['bias']*100:.1f})" if "bias" in row else "")
    print(f"  {pid} ({row['site']}): g1 {g1*100:.2f} g3 {g3*100:.2f} "
          f"energy {row['pred_energy_frac']*100:.1f}%{extra}", flush=True)

if rows:
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    m = st.mean([r["plan_g1"] for r in rows]) * 100
    print(f"\n[{cfg.get('exp_name','?')} m{a.margin}] TRUE mean g1 {m:.2f} over {len(rows)} patients -> {a.out}")
    if "bias" in rows[0]:
        print(f"  old harness would have reported "
              f"{st.mean([r['plan_g1_croppedref'] for r in rows])*100:.2f} "
              f"(+{st.mean([r['bias'] for r in rows])*100:.2f} optimistic)")
