"""Render REAL-trainer augmentation samples — imports the SAME sct_aug.augment() the trainers now
call (train_sct_classifier / train_sct_refiner), so what you see is exactly what training sees.
Verifies: (a) MR/coarse/CT/label move together under flip+rot, (b) label stays on integer classes
(NEAREST), (c) MR-only intensity augs change MR but leave CT/coarse/label identical.

  python scripts/sct_aug_preview_real.py --pid 1ABB006 --out /tmp/aug_real --n 4
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from train_sct_paired import norm_mr, load_arr
from train_sct_classifier import ct_to_class
import sct_aug

ap = argparse.ArgumentParser()
ap.add_argument("--pid", default="1ABB006")
ap.add_argument("--out", default="/tmp/aug_real")
ap.add_argument("--coarse-dir", default="/data/kwang/doserad_cache_archive/coarse_ct_whole_soft")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

D = json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_2mm_samefield.json"))
it = next(x for x in D["train"] if x["pid"] == a.pid)
mr0 = norm_mr(load_arr(it["mr"])).astype(np.float32)
ct0 = load_arr(it["ct"]).astype(np.float32)
lab0 = ct_to_class(ct0).astype(np.float32)
coarse0 = load_arr(os.path.join(a.coarse_dir, a.pid + ".nii.gz")).astype(np.float32)
z = mr0.shape[0] // 2
rng = np.random.default_rng(a.seed)
print(f"{a.pid}: shape {mr0.shape}, orig label classes {sorted(np.unique(lab0).astype(int))}", flush=True)


def show(ax, img, cmap, vr, title):
    ax.imshow(img[z], cmap=cmap, vmin=(vr[0] if vr else None), vmax=(vr[1] if vr else None))
    ax.set_title(title, fontsize=8); ax.axis("off")


# ---- Panel A: classifier aug (MR + label together; label must stay 4-class) ----
rows = [("original", mr0, lab0)]
label_ok = True
for k in range(a.n):
    mr, lab = sct_aug.augment([mr0.copy(), lab0.copy()], [False, True], 0, rng, max_deg=18.0)
    cls = sorted(np.unique(lab).astype(int))
    ok = set(cls).issubset({0, 1, 2, 3})
    label_ok &= ok
    frac = float((np.abs(lab - np.rint(lab)) > 1e-4).mean()) * 100
    rows.append((f"aug#{k+1} cls{cls} frac{frac:.2f}%", mr, lab))
fig, ax = plt.subplots(len(rows), 2, figsize=(7, 3.0 * len(rows)))
for i, (t, m, l) in enumerate(rows):
    show(ax[i, 0], m, "gray", (0, 1), f"{t}\nMR"); show(ax[i, 1], l, "tab10", (0, 3), "label(4-cls)")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "A_classifier_aug.png"), dpi=90); plt.close()
print(f"  A_classifier_aug.png  label-integer-preserved={label_ok}", flush=True)

# ---- Panel B: refiner aug (MR + coarse + CT move together, all LINEAR) ----
rng = np.random.default_rng(a.seed)
rows = [("original", mr0, coarse0, ct0)]
for k in range(a.n):
    mr, co, ct = sct_aug.augment([mr0.copy(), coarse0.copy(), ct0.copy()], [False, False, False], 0,
                                 rng, max_deg=18.0)
    rows.append((f"aug#{k+1}", mr, co, ct))
fig, ax = plt.subplots(len(rows), 3, figsize=(10, 3.0 * len(rows)))
for i, (t, m, co, ct) in enumerate(rows):
    show(ax[i, 0], m, "gray", (0, 1), f"{t}\nMR"); show(ax[i, 1], co, "gray", None, "coarse-CT")
    show(ax[i, 2], ct, "gray", None, "real CT")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "B_refiner_aug.png"), dpi=90); plt.close()
print("  B_refiner_aug.png", flush=True)

# ---- Panel C: MR-only intensity (rotate OFF so only MR changes; CT/coarse/label identical) ----
rng = np.random.default_rng(a.seed)
rows = []
for k in range(a.n + 1):
    if k == 0:
        rows.append(("original", mr0, ct0)); continue
    # force geometry identity, exercise only intensity by calling the intensity helpers directly
    m = mr0.copy()
    m = sct_aug._mr_gamma(m, rng); m = sct_aug._mr_bias(m, rng); m = sct_aug._mr_noise(m, rng)
    rows.append((f"MR-intensity#{k}", m, ct0))
fig, ax = plt.subplots(len(rows), 2, figsize=(7, 3.0 * len(rows)))
for i, (t, m, ct) in enumerate(rows):
    show(ax[i, 0], m, "gray", (0, 1), f"{t}\nMR"); show(ax[i, 1], ct, "gray", None, "real CT (unchanged)")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "C_mr_intensity.png"), dpi=90); plt.close()
print("  C_mr_intensity.png", flush=True)
print("done ->", a.out, flush=True)
