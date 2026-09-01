"""Unified in-body HU-MAE for every sCT-generating model on the 16 Photon-MRI val cases.
Same metric for all (mean-per-patient): all / bone-band (CT>200) / lung-band (CT<-300 in body) / SSIM,
vs the real challenge CT. Static models read precomputed volumes; dose-aware ones run their synth.
    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/measure_sct_hu.py
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np, torch
import SimpleITK as sitk
from train_dose_e2e import E2E, CT_LO, CT_HI
import yaml

RAW = "/data/kwang/DoseRad2026_raw/photon/training"
DEV = "cuda"
VAL = json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"]
COARSE = "/data/kwang/doserad_cache_archive/coarse_ct_v1"

STATIC = {  # label -> dir (resolve <pid>/sCT.mha or <pid>.nii.gz)
    "v4 (paired ResUNet)":   "/data/kwang/sct_eval/v4",
    "STU-Net (paired)":      "/data/kwang/sct_eval/stunet",
    "CycleGAN-v4":           "/data/kwang/sct_eval/cycleganV4_latest",
    "VBoussot (SynthRAD25)": "/data/kwang/synthrad_run/Predictions/Out/Dataset",
    "coarse (classifier)":   COARSE,
    "image refiner":         "/data/kwang/doserad_cache_archive/sct_imgrefiner_val",
}
E2EM = {  # label -> (config, ckpt, coarse?)
    "C lam0.3 (dose-aware)": ("configs/experiments/mri_dose_e2e_C_lam03.yaml",
                              "/home/kaiwang/doserad2026_workdir/runs/mri_dose_e2e_C_frozenv13_lam03/best.pt", False),
    "refiner-da":            ("configs/experiments/mri_dose_e2e_refiner_da.yaml",
                              "/home/kaiwang/doserad2026_workdir/runs/mri_dose_e2e_refiner_da/best.pt", True),
}


def load(p): return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)


def resolve(d, pid):
    for c in (Path(d) / pid / "sCT.mha", Path(d) / f"{pid}.nii.gz", Path(d) / f"{pid}.mha"):
        if c.exists(): return c
    return None


def _g1d(sig=1.5, r=5):
    x = torch.arange(-r, r + 1, dtype=torch.float32); g = torch.exp(-x * x / (2 * sig * sig)); return g / g.sum()


def ssim3d(a, b):
    g = _g1d().to(DEV); ta = torch.as_tensor(a, device=DEV)[None, None]; tb = torch.as_tensor(b, device=DEV)[None, None]
    def blur(t):
        for dim in (2, 3, 4):
            sh = [1, 1, 1, 1, 1]; sh[dim] = g.numel(); pad = [0, 0, 0]; pad[dim - 2] = g.numel() // 2
            t = torch.nn.functional.conv3d(t, g.view(sh), padding=tuple(pad))
        return t
    ma, mb = blur(ta), blur(tb); va = blur(ta * ta) - ma * ma; vb = blur(tb * tb) - mb * mb; vab = blur(ta * tb) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * ma * mb + c1) * (2 * vab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).mean())


def metrics(sct_hu, ct):
    body = ct > -500; bone = ct > 200; lung = body & (ct < -300); d = np.abs(sct_hu - ct)
    def m(msk): return float(d[msk].mean()) if msk.any() else float("nan")
    n = lambda v: np.clip((v + 1000) / 2000, 0, 1)
    return m(body), m(bone), m(lung), ssim3d(n(ct), n(sct_hu))


@torch.no_grad()
def e2e_sct(cfg_path, ckpt, coarse):
    cfg = yaml.safe_load(open(cfg_path)); net = E2E(cfg).to(DEV).eval()
    sd = torch.load(ckpt, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    out = {}
    for pid in VAL:
        a = load(Path(RAW) / pid / "image" / "mr.mha"); lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        mr01 = torch.from_numpy(np.clip((a - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)).to(DEV)
        if coarse:
            cv = load(Path(COARSE) / f"{pid}.nii.gz")
            co = torch.from_numpy(np.clip((cv - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)).to(DEV)
            x = torch.stack([mr01, co], 0)[None]
        else:
            x = mr01[None, None]
        with torch.autocast("cuda"):
            out[pid] = (net.sct01(x)[0, 0] * (CT_HI - CT_LO) + CT_LO).float().cpu().numpy()
    del net; torch.cuda.empty_cache(); return out


def run(label, getter):
    rows = {"all": [], "bone": [], "lung": [], "ssim": []}
    for pid in VAL:
        ct = load(Path(RAW) / pid / "image" / "ct.mha"); sct = getter(pid)
        if sct is None or sct.shape != ct.shape: continue
        a, b, l, s = metrics(sct, ct)
        rows["all"].append(a); rows["bone"].append(b); rows["lung"].append(l); rows["ssim"].append(s)
    M = lambda k: float(np.nanmean(rows[k])) if rows[k] else float("nan")
    print(f"{label:24s} all {M('all'):6.1f} | bone {M('bone'):6.1f} | lung {M('lung'):6.1f} | SSIM {M('ssim'):.3f} (n={len(rows['all'])})")


def main():
    print(f"{'model':24s} {'allHU':>9s} | {'boneHU':>6s} | {'lungHU':>6s} | SSIM  (in-body, 16 val)")
    for label, d in STATIC.items():
        run(label, lambda pid, d=d: (load(resolve(d, pid)) if resolve(d, pid) else None))
    for label, (cfg, ck, co) in E2EM.items():
        cache = e2e_sct(cfg, ck, co)
        run(label, lambda pid, c=cache: c.get(pid))


if __name__ == "__main__":
    main()
