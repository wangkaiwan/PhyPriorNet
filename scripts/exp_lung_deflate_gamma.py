"""Validate the proton-CT LUNG dose-deflation correction on the 71 cached held-out CV predictions.

Correction (from diag_proton_lung_bias.py, 35 lung pts): pred dose is systematically OVER in lung-HU
voxels, growing with dose band: +0.45% (10-30% max), +0.75% (30-60), +1.25% (60-100), ~0 below 10%.
Apply: dose' = dose * f(dose_frac) on voxels with CT < -500 HU, f smoothly interpolated between band
centers; measure plan gamma 1%/1mm before/after per patient.

Overfit guard: band factors are re-fit on ODD-indexed lung patients only; gamma delta is reported
separately for the EVEN-indexed (validation) patients. Abd patients get a no-harm check.

GPU gamma (interp_fraction=10) for screening speed; the winner should be re-verified with pymedphys.
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/exp_lung_deflate_gamma.py [--site lung|abd] [--max N]
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
import numpy as np

PRED_DIR = "/home/kaiwang/doserad2026_workdir/runs/proton_ct_diag_pred"
LUNG_HU = -500.0
BANDS = [(0.02, 0.10), (0.10, 0.30), (0.30, 0.60), (0.60, 1.01)]

ap = argparse.ArgumentParser()
ap.add_argument("--site", default="lung"); ap.add_argument("--max", type=int, default=99)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
lung_files = [f for f in files if "THB" in os.path.basename(f)]
abd_files = [f for f in files if "ABB" in os.path.basename(f)]

# ---- fit band factors on ODD lung patients ----
fit_files = lung_files[1::2]
bsum = np.zeros(len(BANDS)); bcnt = np.zeros(len(BANDS))
for f in fit_files:
    z = np.load(f); pred, gt, ct = z["pred"], z["gt"], z["ct"]
    gmax = float(gt.max()); lung = ct < LUNG_HU
    for i, (lo, hi) in enumerate(BANDS):
        m = lung & (gt >= lo * gmax) & (gt < hi * gmax)
        if m.sum() < 500: continue
        bsum[i] += float((pred[m] * gt[m]).sum() / (gt[m] ** 2).sum()); bcnt[i] += 1
B = np.where(bcnt > 0, bsum / np.maximum(bcnt, 1), 1.0)
print(f"[fit on {len(fit_files)} odd lung pts] band b = " + " ".join(f"{b:.4f}" for b in B), flush=True)
CENTERS = np.array([(lo + hi) / 2 for lo, hi in BANDS])
FACTORS = 1.0 / B     # multiplicative correction per band center

def deflate(pred, ct, planmax):
    frac = np.clip(pred / max(planmax, 1e-9), 0, 1)
    f = np.interp(frac, CENTERS, FACTORS, left=1.0, right=FACTORS[-1]).astype(np.float32)
    lung = ct < LUNG_HU
    out = pred.copy()
    out[lung] = pred[lung] * f[lung]
    return out

def plan_gamma(pred, gt, spacing, rx):
    from doserad.eval.gamma_gpu import gamma_array_gpu
    gmax = float(gt.max())
    zz, yy, xx = np.where(gt >= 0.05 * gmax); m = 4
    cr = (slice(max(int(zz.min())-m,0), int(zz.max())+m+1),
          slice(max(int(yy.min())-m,0), int(yy.max())+m+1),
          slice(max(int(xx.min())-m,0), int(xx.max())+m+1))
    g, msk = gamma_array_gpu(pred[cr], gt[cr], tuple(spacing), gmax, 1.0, 1.0, interp_fraction=10)
    return 100.0 * float((g[msk] <= 1.0).mean()) if msk.any() else float("nan")

todo = (lung_files if a.site == "lung" else abd_files)[:a.max]
print(f"pid      role   before   after   delta", flush=True)
d_fit, d_val = [], []
for idx, f in enumerate(todo):
    pid = os.path.basename(f)[:-4]
    role = ("fit" if (a.site == "lung" and idx % 2 == 1) else ("val" if a.site == "lung" else "abd"))
    z = np.load(f)
    pred, gt, ct, sp = z["pred"], z["gt"], z["ct"], z["spacing"]
    g0 = plan_gamma(pred, gt, sp, float(z["rx"]))
    g1 = plan_gamma(deflate(pred, ct, float(pred.max())), gt, sp, float(z["rx"]))
    (d_fit if role == "fit" else d_val).append(g1 - g0)
    print(f"{pid} {role:4s}  {g0:6.2f}  {g1:6.2f}  {g1-g0:+6.2f}", flush=True)

for name, d in (("FIT(odd)", d_fit), ("VAL" if a.site == "lung" else "ABD-no-harm", d_val)):
    if d:
        d = np.array(d)
        print(f">>> {name}: mean {d.mean():+.3f} +- {d.std():.3f}  (improved {100*(d>0).mean():.0f}% of {len(d)})", flush=True)
