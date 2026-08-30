"""Render sCT from the best A/B and C dose-aware models vs MRI / real CT, with diff maps,
focusing on bone & lung fidelity. Full-volume forward (NOT sliding window — train/test matched).
    conda run -n doserad python scripts/viz_sct_models.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, yaml
from pathlib import Path
from doserad.io.mha import load_mha
from train_dose_e2e import E2E, CT_LO, CT_HI

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
DEV = "cuda"
R = "/home/kaiwang/doserad2026_workdir/runs"
OUT = Path("/home/kaiwang/doserad2026_workdir/sct_viz"); OUT.mkdir(exist_ok=True, parents=True)

MODELS = [
    ("B (joint, 87.6)",   "configs/experiments/mri_dose_e2e8.yaml",        f"{R}/mri_dose_e2e8_doseaware_img/state.pt"),
    ("C lam0.3 (87.0)",   "configs/experiments/mri_dose_e2e_C_lam03.yaml", f"{R}/mri_dose_e2e_C_frozenv13_lam03/best.pt"),
    ("C lam1.0 (HU71)",   "configs/experiments/mri_dose_e2e_C_lam10.yaml", f"{R}/mri_dose_e2e_C_frozenv13_lam10/best.pt"),
]
PATIENTS = ["1THB016", "1THB002", "1ABB006"]


@torch.no_grad()
def gen_sct(cfg_path, ckpt, mr01):
    cfg = yaml.safe_load(open(cfg_path))
    net = E2E(cfg).to(DEV).eval()
    sd = torch.load(ckpt, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    with torch.autocast("cuda"):
        sct = net.sct01(mr01[None, None])[0, 0] * (CT_HI - CT_LO) + CT_LO
    del net; torch.cuda.empty_cache()
    return sct.float().cpu().numpy()


def pick_slice(ct):
    body = ct > -500
    # axial slice with most bone (HU>200) — guarantees bone visible
    bone = (ct > 200) & body
    z = int(np.argmax(bone.sum(axis=(1, 2)))) if bone.any() else ct.shape[0] // 2
    return z


def main():
    # cache sCTs per patient per model
    for pid in PATIENTS:
        mr = load_mha(Path(ROOT) / pid / "image" / "mr.mha"); amr = mr.array.astype(np.float32)
        lo, hi = np.percentile(amr, 1), np.percentile(amr, 99)
        mr01 = torch.from_numpy(np.clip((amr - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)).to(DEV)
        ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha").array.astype(np.float32)
        body = ct > -500
        scts = [(name, gen_sct(cfg, ck, mr01)) for name, cfg, ck in MODELS]
        z = pick_slice(ct)
        ncol = 2 + len(scts)
        fig, ax = plt.subplots(2, ncol, figsize=(2.6 * ncol, 5.4))
        def im(a, img, title, **kw):
            h = a.imshow(img, aspect="equal", **kw); a.set_title(title, fontsize=8.5)
            a.set_xticks([]); a.set_yticks([]); return h
        im(ax[0, 0], amr[z], f"{pid} z={z}\nMRI", cmap="gray")
        im(ax[0, 1], ct[z], "real CT [-1000,1000]", cmap="gray", vmin=-1000, vmax=1000)
        ax[1, 0].axis("off"); ax[1, 1].axis("off")
        ax[1, 1].text(0.5, 0.5, "row 2:\nsCT - CT (HU)\n[-400,400]", ha="center", va="center", fontsize=9)
        for j, (name, s) in enumerate(scts, start=2):
            bone_mae = float(np.abs((s - ct)[(ct > 200) & body]).mean()) if ((ct > 200) & body).any() else 0
            lung_mae = float(np.abs((s - ct)[(ct < -300) & body]).mean()) if ((ct < -300) & body).any() else 0
            im(ax[0, j], s[z], f"{name}\nbone {bone_mae:.0f} / lung {lung_mae:.0f} HU",
               cmap="gray", vmin=-1000, vmax=1000)
            h = im(ax[1, j], np.clip(s[z] - ct[z], -400, 400), "", cmap="seismic", vmin=-400, vmax=400)
        plt.colorbar(h, ax=ax[1, ncol - 1], fraction=0.046, pad=0.02)
        site = "LUNG" if "THB" in pid else "ABDOMEN"
        fig.suptitle(f"sCT comparison — {pid} ({site}).  Top: sCT (bone/lung in-band HU MAE).  "
                     f"Bottom: error vs real CT (red=too dense, blue=too light).", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = OUT / f"sct_cmp_{pid}.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
