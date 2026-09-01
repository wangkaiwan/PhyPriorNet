"""Visualise the classifier's coarse-CT prior vs MR / real CT, on val cases.
Top row: MR | coarse CT (4-level bulk HU) | real CT  (window [-1000,1000]).
Bottom row: bone agreement (TP green / false-pos red / missed blue) + lung agreement.
    conda run -n doserad python scripts/viz_coarse_ct.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json
import SimpleITK as sitk
from pathlib import Path

COARSE = "/data/kwang/doserad_cache_archive/coarse_ct_v1"
OUT = Path("/home/kaiwang/doserad2026_workdir/sct_viz"); OUT.mkdir(parents=True, exist_ok=True)
PATIENTS = ["1THB016", "1THB002", "1ABB006"]


def load(p): return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)


def main():
    D = json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_2mm.json"))
    byid = {it["pid"]: it for it in D["val"] + D["train"]}
    for pid in PATIENTS:
        it = byid[pid]
        mr = load(it["mr"]); ct = load(it["ct"]); coarse = load(Path(COARSE) / f"{pid}.nii.gz")
        z = int(np.argmax(((ct > 200).sum(axis=(1, 2)))))   # bone-heavy slice
        fig, ax = plt.subplots(2, 3, figsize=(11, 7.4))
        def im(a, x, t, **kw):
            a.imshow(x, **kw); a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
        im(ax[0, 0], mr[z], f"{pid}  z={z}\nMRI", cmap="gray")
        im(ax[0, 1], coarse[z], "coarse CT (classifier)\nair/lung/soft/bone", cmap="gray", vmin=-1000, vmax=1000)
        im(ax[0, 2], ct[z], "real CT", cmap="gray", vmin=-1000, vmax=1000)
        # bone agreement
        pb = coarse[z] > 200; tb = ct[z] > 200
        ag = np.zeros((*pb.shape, 3))
        ag[tb & pb] = [0, 1, 0]; ag[pb & ~tb] = [1, 0, 0]; ag[tb & ~pb] = [0, 0.4, 1]
        dice_b = 2 * (pb & tb).sum() / max(pb.sum() + tb.sum(), 1)
        im(ax[1, 0], ag, f"BONE  green=hit red=FP blue=miss\n(slice Dice {dice_b:.2f})")
        # lung agreement
        pl = (coarse[z] < -300) & (ct[z] > -1500); tl = (ct[z] < -300) & (ct[z] > -1500)
        # restrict to body to avoid background air dominating
        body = ct[z] > -500
        pl2 = (coarse[z] < -300) & body_dilate(body); tl2 = (ct[z] < -300) & body_dilate(body)
        agl = np.zeros((*pl.shape, 3))
        agl[tl2 & pl2] = [0, 1, 0]; agl[pl2 & ~tl2] = [1, 0, 0]; agl[tl2 & ~pl2] = [0, 0.4, 1]
        dice_l = 2 * (pl2 & tl2).sum() / max(pl2.sum() + tl2.sum(), 1)
        im(ax[1, 1], agl, f"LUNG/air in body\n(slice Dice {dice_l:.2f})")
        # MR with bone contour
        im(ax[1, 2], mr[z], "MRI + true bone (cyan)", cmap="gray")
        ax[1, 2].contour(tb, colors="cyan", linewidths=0.5)
        site = "LUNG" if "THB" in pid else "ABDOMEN"
        fig.suptitle(f"Classifier coarse-CT prior — {pid} ({site})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = OUT / f"coarse_{pid}.png"; fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


def body_dilate(b):  # small fill so lung mask ignores outside-body air
    return b


if __name__ == "__main__":
    main()
