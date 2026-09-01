"""Cheap IMAGE-LEVEL feasibility test for test-time MR normalization vs the cross-institution shift.
The deployed pipeline only does p1/p99 min-max (_norm_mr01), which does NOT undo the shift's two surviving
components: (a) a low-freq multiplicative BIAS FIELD, (b) a post-norm GAMMA (contrast). Hypothesis:
N4 bias-field correction + histogram-matching to a FIXED reference clean MR undoes both, deterministically,
with NO clean cost. Before spending a dose-gamma run, check in MR (normalized) space:
  clean  vs  shifted(no corr)  vs  shifted+corr   — MAE against _norm_mr01(clean).
If corr MAE << shifted MAE (→ ~clean), the normalization works; then wire it into the dose eval.
Usage: python scripts/prototype_mr_normalize.py
"""
import os, sys, json
import numpy as np, SimpleITK as sitk
from pathlib import Path
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
FROZEN = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))["held16"]
TEST = [p for p in FROZEN if "THB" in p][:3] + [p for p in FROZEN if "ABB" in p][:1]   # 3 lung + 1 abd
REF_PID = [p for p in FROZEN if "ABB" in p][-1]   # a DIFFERENT clean patient = fixed deploy reference


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)


def apply_shift(a, amp=0.5, gam=1.5, seed=0):
    from scipy.ndimage import zoom
    rng = np.random.default_rng(seed)
    Z, Y, X = a.shape
    small = rng.standard_normal((max(Z // 8, 2), max(Y // 8, 2), max(X // 8, 2))).astype(np.float32)
    f = zoom(small, [Z / small.shape[0], Y / small.shape[1], X / small.shape[2]], order=1)
    bias = 1.0 + amp * (f / (np.abs(f).max() + 1e-6))
    ap = a * bias
    lo, hi = np.percentile(ap, 1), np.percentile(ap, 99)
    n = np.clip((ap - lo) / max(hi - lo, 1.0), 0, 1)
    return (np.clip(n, 0, None) ** gam) * (hi if hi > 0 else 1.0)


def n4(a):
    img = sitk.Cast(sitk.GetImageFromArray(a.astype(np.float32)), sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    shrunk = sitk.Shrink(img, [2, 2, 2]); mshr = sitk.Shrink(mask, [2, 2, 2])
    f = sitk.N4BiasFieldCorrectionImageFilter(); f.SetMaximumNumberOfIterations([20, 15, 10])
    f.Execute(shrunk, mshr)
    logbias = f.GetLogBiasFieldAsImage(img)                 # full-res reconstruction
    return sitk.GetArrayFromImage(img / sitk.Exp(logbias)).astype(np.float32)


def histmatch(a, ref):
    src = sitk.GetImageFromArray(a.astype(np.float32)); dst = sitk.GetImageFromArray(ref.astype(np.float32))
    m = sitk.HistogramMatchingImageFilter(); m.SetNumberOfHistogramLevels(256)
    m.SetNumberOfMatchPoints(12); m.ThresholdAtMeanIntensityOn()
    return sitk.GetArrayFromImage(m.Execute(src, dst)).astype(np.float32)


ref_mr = sitk.GetArrayFromImage(sitk.ReadImage(f"{PROT}/{REF_PID}/image/mr.mha")).astype(np.float32)
print(f"reference (clean) = {REF_PID}\n{'pid':10} {'shifted':>9} {'N4only':>9} {'N4+hist':>9} {'histonly':>9}  (MAE vs clean, norm space)", flush=True)
for pid in TEST:
    a = sitk.GetArrayFromImage(sitk.ReadImage(f"{PROT}/{pid}/image/mr.mha")).astype(np.float32)
    clean_n = norm01(a)
    sh = apply_shift(a)
    mae = lambda x: float(np.abs(norm01(x) - clean_n).mean())
    sh_mae = mae(sh); n4_mae = mae(n4(sh)); n4h_mae = mae(histmatch(n4(sh), ref_mr)); h_mae = mae(histmatch(sh, ref_mr))
    best = min(n4_mae, n4h_mae, h_mae)
    print(f"{pid:10} {sh_mae:9.4f} {n4_mae:9.4f} {n4h_mae:9.4f} {h_mae:9.4f}   {'RECOVERS' if best < 0.7*sh_mae else 'weak/worse'}", flush=True)
print(">>> want a correction column << shifted. N4-only targets the spatial bias (the CLF-relevant part).", flush=True)
