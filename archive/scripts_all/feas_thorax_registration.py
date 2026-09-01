"""Feasibility: do the thorax MR-CT training pairs have residual breathing MISALIGNMENT that a deformable
registration can fix? The pairs are co-registered on one grid but docs note body Dice thorax 0.87 vs abd 0.99.
If real, the sCT learns a blurred lung density (input MR displaced from label CT) -> lung range error ->
lung gamma fails (our worst region). Measure, per pair: lung-mask Dice (CT air vs MR dark-lung) BEFORE vs
AFTER a Mattes-MI BSpline deformable reg (CT->MR), + the deformation-field magnitude (mm) inside the body.
Big before->after Dice gain + non-trivial deformation = real, correctable misalignment -> worth the pipeline.
Usage: python scripts/feas_thorax_registration.py
"""
import numpy as np, SimpleITK as sitk, json
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
FROZEN = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))["held16"]
PIDS = [p for p in FROZEN if "THB" in p][:3] + [p for p in FROZEN if "ABB" in p][:1]


def body_mask(img):  # Otsu foreground, largest component, filled
    m = sitk.OtsuThreshold(img, 0, 1) == 0 if False else (sitk.OtsuThreshold(img, 0, 1))
    m = sitk.BinaryMorphologicalClosing(m, [3, 3, 3])
    return sitk.BinaryFillhole(m)


def lung_ct(ct, body):   # air inside body
    a = sitk.GetArrayFromImage(ct); b = sitk.GetArrayFromImage(body).astype(bool)
    return ((a < -300) & b).astype(np.uint8)


def lung_mr(mr, body):   # dark region inside body, keep 2 largest INTERIOR components (exclude boundary/skin)
    from scipy.ndimage import binary_erosion, label
    a = sitk.GetArrayFromImage(mr); b = sitk.GetArrayFromImage(body).astype(bool)
    interior = binary_erosion(b, iterations=3)          # drop skin/boundary dark rim
    inside = a[interior]
    thr = np.percentile(inside, 25) if inside.size else 0
    dark = (a < thr) & interior
    lab, n = label(dark)
    if n == 0: return dark.astype(np.uint8)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.argsort(sizes)[-2:]                        # 2 largest = the lungs
    return np.isin(lab, keep).astype(np.uint8)


def dice(x, y):
    x = x.astype(bool); y = y.astype(bool); s = x.sum() + y.sum()
    return 2.0 * (x & y).sum() / s if s else 1.0


def deformable(fixed, moving):   # Mattes-MI BSpline, CT(moving)->MR(fixed)
    f = sitk.Cast(fixed, sitk.sitkFloat32); m = sitk.Cast(moving, sitk.sitkFloat32)
    tx = sitk.BSplineTransformInitializer(f, [6, 6, 4])
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.05, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=50)
    R.SetInitialTransform(tx, inPlace=True)
    R.SetShrinkFactorsPerLevel([2, 1]); R.SetSmoothingSigmasPerLevel([1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R.Execute(f, m)


print(f"{'pid':10} {'site':4} {'lungDice_before':>15} {'lungDice_after':>15} {'defmag_LUNG_p95':>15}", flush=True)
for pid in PIDS:
    site = "lung" if "THB" in pid else "abd"
    mr = sitk.ReadImage(f"{PROT}/{pid}/image/mr.mha"); ct = sitk.ReadImage(f"{PROT}/{pid}/image/ct.mha")
    mr = sitk.Shrink(mr, [4, 4, 2]); ct = sitk.Shrink(ct, [4, 4, 2])   # coarse: breathing misalign is low-freq
    bmask = body_mask(mr)
    lct = lung_ct(ct, bmask); lmr = lung_mr(mr, bmask)
    d_before = dice(lct, lmr)
    tx = deformable(mr, ct)
    ct_w = sitk.Resample(ct, mr, tx, sitk.sitkLinear, -1000.0)
    lct_w = lung_ct(ct_w, bmask)
    d_after = dice(lct_w, lmr)
    disp = sitk.GetArrayFromImage(sitk.TransformToDisplacementField(tx, sitk.sitkVectorFloat32,
        mr.GetSize(), mr.GetOrigin(), mr.GetSpacing(), mr.GetDirection()))
    magf = np.linalg.norm(disp, axis=-1)
    ll = lct.astype(bool)
    dm_lung = np.percentile(magf[ll], 95) if ll.any() else 0.0
    print(f"{pid:10} {site:4} {d_before:15.3f} {d_after:15.3f} {dm_lung:15.2f}", flush=True)
print(">>> lung Dice jump (before->after) + defmag_LUNG p95 several mm = real breathing misalignment, fixable.", flush=True)
