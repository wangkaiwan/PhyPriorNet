"""Diagnose the MRI-sCT cross-institution generalisation gap (internal held-out ~87 vs leaderboard 76,
same ~12 gap on BOTH photon-MRI and proton-MRI; CT has none). Hypothesis: the leaderboard's test MR
comes from a different 0.35T ViewRay unit/institution (LMU vs AUMC) with a different intensity/contrast
profile, and our sCT pipeline (clf coarse + synth) is not robust to it. Our internal eval hides this by
(a) same-cohort held-out and (b) using PRECOMPUTED coarse instead of the container's LIVE clf.

This runs the EXACT container path (container.mri_synth.synth_density = live clf + synth) on the held-out
16, WITH vs WITHOUT a realistic intensity/contrast shift that SURVIVES the percentile-1/99 norm (a smooth
bias field + a post-norm gamma — a global scale would just be normalised away). Reports how much the
sCT density changes, and how much the 4-class tissue coarse (bone/lung/soft) flips. A large change =>
the pipeline is intensity-fragile => the shift explains the leaderboard gap => MR-intensity aug is the fix.

  CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/mri_shift_robustness.py --n 8
"""
from __future__ import annotations
import argparse, json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch, SimpleITK as sitk
from scipy.ndimage import gaussian_filter, zoom
from container.mri_synth import load_classifier, _norm_mr01, REP_HU, CT_LO, CT_HI, _pad16, DENS_MAX
import torch.nn.functional as F

BANDS = np.array([-700., -300., 200.])   # air/lung/soft/bone edges (same as clf)
def to_cls(hu): return np.searchsorted(BANDS, hu)

def bias_field(shape, rng, amp):
    small = rng.normal(0, 1, [max(s // 8, 2) for s in shape]).astype(np.float32)
    small = gaussian_filter(small, 2.0)
    f = zoom(small, [shape[i] / small.shape[i] for i in range(3)], order=1)
    return 1.0 + amp * (f / (np.abs(f).max() + 1e-6))

@torch.no_grad()
def coarse_and_sct(a_mr, clf, synth, scfg_dd, dev):
    """container path: MR array -> live clf coarse -> synth sct01 -> (coarse_hu, sct_hu). 2mm grid."""
    mr01 = torch.from_numpy(_norm_mr01(a_mr)).to(dev)
    mrx = torch.from_numpy(np.transpose(_norm_mr01(a_mr), (2, 1, 0)))[None, None].to(dev)
    with torch.autocast("cuda"):
        X, Y, Z = mrx.shape[-3:]
        xp = F.pad(mrx, (0, _pad16(Z), 0, _pad16(Y), 0, _pad16(X)))
        p = torch.softmax(clf(xp)[..., :X, :Y, :Z].float(), 1)[0]
        coarse = (p * torch.from_numpy(REP_HU).to(dev).view(-1, 1, 1, 1)).sum(0).permute(2, 1, 0)
        co = torch.clamp((coarse - CT_LO) / (CT_HI - CT_LO), 0, 1)
        sct01 = synth.sct01(torch.stack([mr01, co], 0)[None])[0, 0]
    sct_hu = (sct01.float().clamp(0, 1) * (CT_HI - CT_LO) + CT_LO).cpu().numpy()
    return coarse.cpu().numpy(), sct_hu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--clf", default="/data/kwang/sct_classify_runs/clf_whole/best.pt")
    ap.add_argument("--synth-config", default="configs/experiments/all75/all75_r3_protonmri.yaml")
    ap.add_argument("--synth-ckpt", default="/home/kaiwang/doserad2026_workdir/runs/all75_r3_protonmri/state.pt")
    ap.add_argument("--bias", type=float, default=0.25)   # ViewRay-unit bias-field amplitude
    ap.add_argument("--gamma", type=float, default=1.25)  # post-norm contrast (survives pct norm)
    a = ap.parse_args()
    dev = "cuda"
    import yaml
    from train_dose_e2e import E2E
    scfg = yaml.safe_load(open(a.synth_config))
    net = E2E(scfg).to(dev).eval()
    sd = torch.load(a.synth_ckpt, map_location=dev); net.load_state_dict(sd.get("ema", sd.get("model")))
    clf = load_classifier(a.clf, dev)

    val = json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"][:a.n]
    ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
    rng = np.random.default_rng(0)
    dens_mae, sct_mae, bone_flip, lung_flip = [], [], [], []
    print(f"MR intensity-shift robustness (container path, live clf), bias {a.bias} gamma {a.gamma}", flush=True)
    for pid in val:
        a_mr = sitk.GetArrayFromImage(sitk.ReadImage(f"{ROOT}/{pid}/image/mr.mha")).astype(np.float32)
        # perturbed MR = bias field * MR, then post-norm gamma (survives pct 1/99)
        pf = bias_field(a_mr.shape, rng, a.bias)
        a_pert = a_mr * pf
        c0, s0 = coarse_and_sct(a_mr, clf, net, scfg, dev)      # clean
        # gamma applied to the normalised MR to change contrast (survives pct norm)
        def gamma_mr(arr):
            n = _norm_mr01(arr); ng = np.clip(n, 0, None) ** a.gamma
            return ng * (arr.max() if arr.max() > 0 else 1.0)   # rescale back so pct-norm re-applies
        c1, s1 = coarse_and_sct(gamma_mr(a_pert), clf, net, scfg, dev)   # perturbed
        body = s0 > -500
        dm = np.abs(s1 - s0)[body].mean(); sct_mae.append(dm)
        cl0, cl1 = to_cls(c0), to_cls(c1)
        bone_flip.append(((cl0 == 3) != (cl1 == 3))[body].mean() * 100)
        lung_flip.append(((cl0 == 1) != (cl1 == 1))[body].mean() * 100)
        # density (proton) change
        d0 = np.clip(s0, CT_LO, CT_HI); d1 = np.clip(s1, CT_LO, CT_HI)
        print(f"  {pid}: sCT ΔHU {dm:5.1f} | bone-flip {bone_flip[-1]:4.1f}% | lung-flip {lung_flip[-1]:4.1f}%", flush=True)
    print(f"\nMEAN under shift: sCT ΔHU {np.mean(sct_mae):.1f} | bone class-flip {np.mean(bone_flip):.1f}% "
          f"| lung class-flip {np.mean(lung_flip):.1f}%", flush=True)
    print("(large ΔHU / class-flip => pipeline is intensity-fragile => leaderboard gap is the shift => aug is the fix)", flush=True)

if __name__ == "__main__":
    main()
