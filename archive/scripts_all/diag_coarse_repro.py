"""Diagnose WHY the container live-clf coarse != precomputed coarse_ct_1x1x3_samefield_soft on lung.
For 2 abd + 2 lung pids, compute the coarse three ways and diff vs the precomputed .nii.gz:
  (A) WHOLE-IMAGE clf  (what container/mri_synth does)
  (B) SLIDING-WINDOW clf (precompute_coarse_ct.py DEFAULT)
Try clf = best.pt and last.pt. Whichever (recipe,ckpt) gives ~0 HU diff is what generated the coarse.
Usage: CUDA_VISIBLE_DEVICES=1 python scripts/diag_coarse_repro.py
"""
import os, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, SimpleITK as sitk
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from train_sct_paired import norm_mr, load_arr
from train_sct_classifier import model as clf_model

REP = np.asarray([-1000., -600., 30., 700.], np.float32)
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
PRE = "/data/kwang/coarse_ct_1x1x3_samefield_soft"
PIDS = ["1ABB006", "1ABB161", "1THB002", "1THB016"]   # 2 abd (good) + 2 lung (bad, incl the 48.7)
DEV = "cuda"
pad16 = lambda n: (16 - n % 16) % 16

def load_clf(p):
    net = clf_model(len(REP)).to(DEV).eval()
    net.load_state_dict(torch.load(p, map_location=DEV)["net"]); return net

@torch.no_grad()
def coarse(net, pid, whole):
    mr = torch.from_numpy(norm_mr(load_arr(f"{PROT}/{pid}/image/mr.mha")).astype(np.float32))[None, None].to(DEV)  # (1,1,x,y,z)
    with torch.autocast("cuda"):
        if whole:
            X, Y, Z = mr.shape[-3:]
            xp = F.pad(mr, (0, pad16(Z), 0, pad16(Y), 0, pad16(X)))
            logits = net(xp)[..., :X, :Y, :Z]
        else:
            logits = sliding_window_inference(mr, (128, 128, 128), 4, net, overlap=0.25, mode="gaussian")
    p = torch.softmax(logits.float(), 1)[0]
    c = (p * torch.from_numpy(REP).to(DEV).view(-1, 1, 1, 1)).sum(0).cpu().numpy()   # (x,y,z)
    return np.transpose(c, (2, 1, 0))                                                # -> (z,y,x) like the .nii.gz

for ckpt in ["best.pt", "last.pt"]:
    net = load_clf(f"/data/kwang/sct_classify_runs/clf_1x1x3_samefield/{ckpt}")
    print(f"\n===== clf = clf_1x1x3_samefield/{ckpt} =====", flush=True)
    for pid in PIDS:
        pre = sitk.GetArrayFromImage(sitk.ReadImage(f"{PRE}/{pid}.nii.gz")).astype(np.float32)   # (z,y,x)
        for whole in (True, False):
            c = coarse(net, pid, whole)
            d = np.abs(c - pre)
            print(f"  {pid} {'WHOLE ' if whole else 'SLIDE '}: max|ΔHU| {d.max():7.1f}  mean|ΔHU| {d.mean():6.2f}", flush=True)
