"""Quantify the proton-CT LUNG over-prediction (measurement pass, no gamma yet) on the 71 cached
held-out CV predictions (runs/proton_ct_diag_pred/*.npz: pred/gt/ct/spacing/rx).

The confirmed failure mode [[protonct-lung-overprediction-confirmed]]: over-prediction in the
low-dose (~15% of max) low-gradient scatter tail inside LUNG (low HU). Before designing a fix,
measure the multiplicative bias b = sum(pred*gt)/sum(gt^2) conditioned on (lung HU) x (dose band),
per patient — magnitude, consistency, and which bands carry it.

Usage: python scripts/diag_proton_lung_bias.py [--site lung|abd|all]
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np

PRED_DIR = "/home/kaiwang/doserad2026_workdir/runs/proton_ct_diag_pred"
ap = argparse.ArgumentParser(); ap.add_argument("--site", default="lung"); a = ap.parse_args()

BANDS = [(0.02, 0.10), (0.10, 0.30), (0.30, 0.60), (0.60, 1.01)]   # fractions of plan max (GT)
LUNG_HU = -500.0

files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
rows = []
hdr = "pid      site " + " ".join(f"b[{lo:.2f}-{hi:.2f}]" for lo, hi in BANDS) + "   lungvox%"
print(hdr, flush=True)
for f in files:
    pid = os.path.basename(f)[:-4]
    site = "lung" if "THB" in pid else "abd"
    if a.site != "all" and site != a.site: continue
    z = np.load(f)
    pred = z["pred"]; gt = z["gt"]; ct = z["ct"]
    gmax = float(gt.max())
    lung = ct < LUNG_HU
    bs = []
    for lo, hi in BANDS:
        m = lung & (gt >= lo * gmax) & (gt < hi * gmax)
        if m.sum() < 500:
            bs.append(np.nan); continue
        bs.append(float((pred[m] * gt[m]).sum() / (gt[m] ** 2).sum()))
    lv = 100.0 * float((lung & (gt >= 0.02 * gmax)).sum()) / max(float((gt >= 0.02 * gmax).sum()), 1)
    rows.append((pid, site, bs, lv))
    print(f"{pid} {site:4s} " + " ".join(f"{b:12.4f}" if b == b else f"{'--':>12s}" for b in bs) + f"   {lv:6.1f}", flush=True)

print("\n=== per-band mean +- std (site=%s, n=%d) ===" % (a.site, len(rows)))
for i, (lo, hi) in enumerate(BANDS):
    v = np.array([r[2][i] for r in rows if r[2][i] == r[2][i]])
    if len(v):
        over = 100.0 * float((v > 1.0).mean())
        print(f"  band {lo:.2f}-{hi:.2f}: b {v.mean():.4f} +- {v.std():.4f}  ({over:.0f}% of patients over 1)")
print("READ: consistent b>1 in the low bands (0.02-0.30) inside lung = the correctable over-prediction;")
print("      the correction factor per band ~= 1/b. Check abd with --site abd (should be ~1 = leave alone).")
