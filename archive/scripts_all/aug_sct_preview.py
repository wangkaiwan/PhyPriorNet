"""Preview & VERIFY sCT-front-end augmentations before they touch training (user gate 2026-07-27).

Standalone — does NOT modify train_sct_refiner/classifier. Loads one paired case (MR + coarse-CT +
real CT + tissue labels), applies each proposed augmentation, and renders before/after panels so the
user can confirm: (a) MR / coarse / CT / label all transform TOGETHER, (b) the discrete LABEL uses
NEAREST interpolation (no class blending), (c) MR intensity aug stays realistic and does NOT touch GT.

  python scripts/aug_sct_preview.py --pid 1ABB006 --out /tmp/aug_preview
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import rotate as ndrotate, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from train_sct_paired import norm_mr, load_arr                    # noqa: E402
from train_sct_classifier import ct_to_class                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pid", default="1ABB006")
ap.add_argument("--out", default="/tmp/aug_preview")
ap.add_argument("--coarse-dir", default="/data/kwang/doserad_cache_archive/coarse_ct_whole_soft")
ap.add_argument("--rot-deg", type=float, default=18.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

D = json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_2mm.json"))
it = next(x for x in D["train"] + D["val"] if x["pid"] == a.pid)
mr = norm_mr(load_arr(it["mr"])).astype(np.float32)               # MR, [0,1]-ish
ct_hu = load_arr(it["ct"]).astype(np.float32)                    # real CT (HU)
coarse = load_arr(os.path.join(a.coarse_dir, a.pid + ".nii.gz")).astype(np.float32)
lab = ct_to_class(ct_hu).astype(np.int16)                        # 4-class tissue label (discrete)
z = mr.shape[0] // 2                                             # a mid axial slice for display
print(f"{a.pid}: shape {mr.shape}, labels {sorted(np.unique(lab))}", flush=True)


# ---- geometric augs: applied IDENTICALLY to MR/coarse/CT, NEAREST for the label ----
def flip3(vol, order): return vol[::order[0], ::order[1], ::order[2]].copy()


def rot_axial(vol, deg, nearest):
    order = 0 if nearest else 1                                  # 0 = nearest (labels), 1 = linear
    return ndrotate(vol, deg, axes=(1, 2), reshape=False, order=order,
                    mode="constant", cval=0.0, prefilter=not nearest)


# ---- MR-only intensity augs (do NOT touch CT/coarse/label) ----
def mr_gamma(m, g): return np.clip(m, 0, None) ** g
def mr_bias(m, amp, sigma=40):
    r = np.random.default_rng(0).normal(0, 1, m.shape).astype(np.float32)
    field = 1.0 + amp * (gaussian_filter(r, sigma) / (np.abs(gaussian_filter(r, sigma)).max() + 1e-6))
    return m * field
def mr_noise(m, s): return m + np.random.default_rng(1).normal(0, s, m.shape).astype(np.float32)


def panel(fname, rows):
    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1: ax = ax[None]
    cols = ["MR", "coarse-CT", "real CT", "label (4-class)"]
    for i, (title, m, co, c, lb) in enumerate(rows):
        for j, (img, cmap, vr) in enumerate([(m, "gray", (0, 1)), (co, "gray", None),
                                             (c, "gray", None), (lb, "tab10", (0, 3))]):
            ax[i, j].imshow(img[z], cmap=cmap, vmin=(vr[0] if vr else None),
                            vmax=(vr[1] if vr else None))
            ax[i, j].set_title(f"{title}\n{cols[j]}" if i == 0 or j == 0 else "", fontsize=8)
            ax[i, j].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, fname), dpi=90); plt.close()
    print(f"  wrote {fname}", flush=True)


# 1) geometric: original vs flip vs rotate — all 4 volumes must move together
panel("1_geometric.png", [
    ("original", mr, coarse, ct_hu, lab),
    ("flip z+y+x", flip3(mr, (-1, -1, -1)), flip3(coarse, (-1, -1, -1)),
     flip3(ct_hu, (-1, -1, -1)), flip3(lab, (-1, -1, -1))),
    (f"rot {a.rot_deg:.0f} (lin MR/CT, NEAREST lab)", rot_axial(mr, a.rot_deg, False),
     rot_axial(coarse, a.rot_deg, False), rot_axial(ct_hu, a.rot_deg, False),
     rot_axial(lab.astype(np.float32), a.rot_deg, True).astype(np.int16)),
])

# 2) LABEL interpolation check: nearest (correct) vs linear (WRONG — blends classes)
lab_near = rot_axial(lab.astype(np.float32), a.rot_deg, True)
lab_lin = rot_axial(lab.astype(np.float32), a.rot_deg, False)     # linear = wrong on purpose, to show
newcls_near = sorted(np.unique(np.rint(lab_near)).astype(int))
newcls_lin = sorted(np.unique(np.rint(lab_lin * 0 + lab_lin)).astype(int))
frac_frac = float((np.abs(lab_lin - np.rint(lab_lin)) > 0.05).mean()) * 100
print(f"  LABEL after NEAREST rot: classes {newcls_near} (must == original {sorted(np.unique(lab))})",
      flush=True)
print(f"  LABEL after LINEAR rot: {frac_frac:.1f}% voxels became fractional (blended) classes -> WRONG",
      flush=True)

# 3) MR intensity augs — MR changes, CT/label unchanged
panel("3_mr_intensity.png", [
    ("original", mr, coarse, ct_hu, lab),
    ("MR gamma 0.7", mr_gamma(mr, 0.7), coarse, ct_hu, lab),
    ("MR bias field", mr_bias(mr, 0.25), coarse, ct_hu, lab),
    ("MR + noise 0.03", mr_noise(mr, 0.03), coarse, ct_hu, lab),
])
print("done ->", a.out, flush=True)
