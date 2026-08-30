"""Localise WHERE and WHY proton-CT loses plan gamma — diagnosis, not a fix.

Unlike the proton-MRI sCT diagnosis, proton-CT uses the REAL CT, so any gamma failure is the DOSE
net's prediction vs the MC ground truth — no synthesis error in the loop. We are 95.30 vs Zimty 96.42
(1.12 gap); this asks where that 1.12 (and our ~4.7 to 100) actually lives.

Method: a first-order LOCAL gamma (1%/1mm), the same functional the challenge scores, but kept
per-voxel and decomposable rather than min-searched:

    gamma(r) ~= |Dpred - Dref| / sqrt( (dose_pct*Dref_local)^2 + (dist_mm*|grad Dref|)^2 )

evaluated on voxels above cutoff*planmax. The denominator's TWO terms are the whole point:
  - dose term  (dose_pct*Dref): a voxel that fails even with a perfect position is DOSE-limited.
  - dist term  (dist_mm*|grad|): a voxel rescued by a <=1mm shift is DISTANCE/RANGE-limited
    (for protons, the distal Bragg falloff — a range error).
Splitting failures into these two buckets says whether the lever is dose-magnitude accuracy or
range/position accuracy, and the SIGN of (pred-gt) at failures says whether it is a systematic bias
(correctable) or scatter (not).

Per-patient rows appended to CSV as computed (resumable). Run after proton_ct_diag_predict.sh:
  python scripts/diag_protonct_gamma.py --pred-dir <dir> --out <csv>
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir", required=True, help="dir of per-patient npz {pred,gt,ct,spacing,rx}")
ap.add_argument("--out", required=True)
ap.add_argument("--cutoff", type=float, default=0.10, help="gamma eval region = > cutoff*planmax")
ap.add_argument("--dose-pct", type=float, default=0.01)
ap.add_argument("--dist-mm", type=float, default=1.0)
a = ap.parse_args()

DOSE_PCT, DIST_MM, CUT = a.dose_pct, a.dist_mm, a.cutoff


def grad_mag(d, spacing):
    """|grad D| in dose/mm on the (z,y,x) grid; spacing is (sx,sy,sz) in mm (sitk order)."""
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    gz, gy, gx = np.gradient(d, sz, sy, sx)          # axis0=z(sz), axis1=y(sy), axis2=x(sx)
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def analyse(f):
    z = np.load(f)
    pred, gt, spacing = z["pred"].astype(np.float32), z["gt"].astype(np.float32), z["spacing"]
    ct = z["ct"].astype(np.float32) if "ct" in z else None
    pmax = float(gt.max())
    if pmax <= 0:
        return None
    region = gt > CUT * pmax                          # local-gamma eval region
    diff = pred - gt
    g = grad_mag(gt, spacing)
    dose_term = DOSE_PCT * gt                          # local %-of-dose criterion (challenge = local)
    dist_term = DIST_MM * g
    denom = np.sqrt(dose_term ** 2 + dist_term ** 2)
    denom = np.maximum(denom, 1e-9)
    gamma = np.abs(diff) / denom                       # first-order local gamma
    inr = region
    fail = inr & (gamma > 1.0)
    n_in, n_fail = int(inr.sum()), int(fail.sum())
    if n_in == 0:
        return None
    passrate = 100.0 * (1 - n_fail / n_in)

    # decomposition of FAILING voxels
    # dose-limited: |diff| already exceeds the dose criterion alone (a perfect shift can't save it)
    dose_limited = fail & (np.abs(diff) > dose_term)
    dist_limited = fail & ~dose_limited                # rescued in principle by <=1mm shift
    # sign of the error at failures: systematic over/under-prediction?
    fdiff = diff[fail]
    frac_over = float((fdiff > 0).mean()) if n_fail else float("nan")
    # where in the dose range do failures sit (fraction of planmax)
    flevel = (gt[fail] / pmax) if n_fail else np.array([0.0])
    # gradient at failures vs overall region (are failures on steep falloff/penumbra?)
    g_region_med = float(np.median(g[inr]))
    g_fail_med = float(np.median(g[fail])) if n_fail else float("nan")
    site = "lung" if Path(f).stem.startswith("1THB") else "abdomen"
    return dict(
        patient=Path(f).stem, site=site, passrate=passrate, n_in=n_in, n_fail=n_fail,
        frac_dose_limited=float(dose_limited.sum()) / max(n_fail, 1),
        frac_dist_limited=float(dist_limited.sum()) / max(n_fail, 1),
        frac_over=frac_over,
        fail_dose_med=float(np.median(flevel)), fail_dose_hi=float(np.mean(flevel > 0.9)),
        grad_fail_over_region=g_fail_med / max(g_region_med, 1e-9),
        mean_signed_diff_at_fail=float(fdiff.mean()) if n_fail else 0.0,
        planmax=pmax,
    )


files = sorted(glob.glob(os.path.join(a.pred_dir, "*.npz")))
out = Path(a.out)
done = set()
cols = ["patient", "site", "passrate", "n_in", "n_fail", "frac_dose_limited", "frac_dist_limited",
        "frac_over", "fail_dose_med", "fail_dose_hi", "grad_fail_over_region",
        "mean_signed_diff_at_fail", "planmax"]
if out.exists():
    done = {ln.split(",")[0] for ln in out.read_text().splitlines()[1:] if ln.strip()}
else:
    out.write_text(",".join(cols) + "\n")

print(f"analysing {len(files)} patients (cutoff {CUT}, {DOSE_PCT*100:.0f}%/{DIST_MM:.0f}mm)", flush=True)
for f in files:
    if Path(f).stem in done:
        continue
    r = analyse(f)
    if r is None:
        continue
    with open(out, "a") as fh:
        fh.write(",".join(f"{r[c]:.5g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"  {r['patient']} ({r['site']}): pass {r['passrate']:.1f} | "
          f"fails: dose-lim {100*r['frac_dose_limited']:.0f}% dist-lim {100*r['frac_dist_limited']:.0f}% | "
          f"over {100*r['frac_over']:.0f}% | fail@grad {r['grad_fail_over_region']:.1f}x region | "
          f"signed {r['mean_signed_diff_at_fail']:+.4f}", flush=True)
print(f"done -> {out}", flush=True)
