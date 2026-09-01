"""Measure clf forward time: WHOLE-IMAGE vs SLIDING-WINDOW on native 1x1x3 volumes (proton-MRI).
clf runs ONCE per source image. Reports seconds each (warm, averaged) so we know the runtime cost of
the mandatory sliding-window recipe. Usage: CUDA_VISIBLE_DEVICES=1 python scripts/time_clf_recipe.py"""
import os, sys, time
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, SimpleITK as sitk, torch.nn.functional as F
from monai.inferers import sliding_window_inference
from train_sct_paired import norm_mr, load_arr
from train_sct_classifier import model as clf_model
REP = np.asarray([-1000.,-600.,30.,700.], np.float32); DEV="cuda"
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
PIDS = ["1ABB161", "1THB002", "1THB191"]   # abd + 2 lung (big volumes)
pad16 = lambda n:(16-n%16)%16
net = clf_model(len(REP)).to(DEV).eval()
net.load_state_dict(torch.load("/data/kwang/sct_classify_runs/clf_1x1x3_samefield/best.pt", map_location=DEV)["net"])

@torch.no_grad()
def run(mrx, whole):
    with torch.autocast("cuda"):
        if whole:
            X,Y,Z = mrx.shape[-3:]; xp = F.pad(mrx,(0,pad16(Z),0,pad16(Y),0,pad16(X)))
            _ = torch.softmax(net(xp)[...,:X,:Y,:Z].float(),1)
        else:
            _ = torch.softmax(sliding_window_inference(mrx,(128,128,128),4,net,overlap=0.25,mode="gaussian").float(),1)
    torch.cuda.synchronize()

for pid in PIDS:
    a = norm_mr(load_arr(f"{PROT}/{pid}/image/mr.mha")).astype(np.float32)
    mrx = torch.from_numpy(a)[None,None].to(DEV)
    for whole in (True, False):
        run(mrx, whole)  # warm
        t=[]
        for _ in range(3):
            t0=time.time(); run(mrx, whole); t.append(time.time()-t0)
        print(f"  {pid} shape{tuple(mrx.shape[-3:])} {'WHOLE ' if whole else 'SLIDE '}: {np.mean(t):.2f}s", flush=True)
