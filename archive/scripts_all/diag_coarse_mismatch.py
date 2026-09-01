"""Decisive check for the whole-image sCT 'catastrophic failure': does the sCT gamma test's ON-THE-FLY
coarse (clf sliding-window + REP_HU, as in eval_sct_gamma_test.new_sct_density) MATCH the PRECOMPUTED
coarse the refiner ACTUALLY trained on (coarse_ct_1x1x3_samefield_whole_soft)? A HU-MAE-53 refiner giving
0.0 dose gamma smells like a coarse-recipe mismatch (wrong clf / whole-vs-sliding / rep_hu). If the two
coarses differ a lot, the test fed the refiner the wrong coarse -> the whole-image sCT may be FINE and
Step 4 was killed on a buggy test. Runs on 2 TRAINING patients (which have a precomputed coarse).
Usage: CUDA_VISIBLE_DEVICES=1 python scripts/diag_coarse_mismatch.py
"""
import os, sys, json
from pathlib import Path
os.environ["TORCHDYNAMO_DISABLE"] = "1"
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
import numpy as np, torch, SimpleITK as sitk
from monai.inferers import sliding_window_inference
import train_sct_classifier as CLF
from train_sct_paired import norm_mr, load_arr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CT_LO, CT_HI = -1000.0, 2000.0
REP_HU = np.asarray([-1000., -600., 30., 700.], np.float32)          # as hardcoded in eval_sct_gamma_test
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
PHOTON = "/data/kwang/DoseRad2026_raw/photon/training"
PRECOMP = "/data/kwang/coarse_ct_1x1x3_samefield_whole_soft"          # what the refiner trained on
CLF_CKPT = os.environ.get("DOSERAD_NEW_CLF", "/data/kwang/sct_classify_runs/clf_1x1x3_samefield_whole/best.pt")
PIDS = ["1ABB006", "1ABB030"]

clf = CLF.model().to(DEV).eval(); clf.load_state_dict(torch.load(CLF_CKPT, map_location="cpu")["net"])
print(f"[coarse-diff] clf={CLF_CKPT}\n  REP_HU={REP_HU.tolist()}  precomp={PRECOMP}", flush=True)

def mr_of(pid):
    for root in (PROT, PHOTON):
        p = f"{root}/{pid}/image/mr.mha"
        if os.path.exists(p): return norm_mr(load_arr(p)).astype(np.float32)   # (x,y,z), same as training
    raise FileNotFoundError(pid)

@torch.no_grad()
def test_coarse_hu(mr):
    x = torch.from_numpy(mr)[None, None].to(DEV)
    with torch.autocast("cuda", enabled=(DEV == "cuda")):
        logit = sliding_window_inference(x, (128, 128, 128), 4, clf, overlap=0.25, mode="gaussian")
    p = torch.softmax(logit.float(), 1)[0]
    return (p * torch.from_numpy(REP_HU).to(DEV).view(-1, 1, 1, 1)).sum(0).cpu().numpy()   # HU (x,y,z)

for pid in PIDS:
    mr = mr_of(pid)
    test_hu = test_coarse_hu(mr)                                       # (x,y,z)
    pre = load_arr(f"{PRECOMP}/{pid}.nii.gz").astype(np.float32)       # (x,y,z), same loader as training
    if pre.shape != test_hu.shape:
        print(f"  {pid}: SHAPE MISMATCH pre {pre.shape} vs test {test_hu.shape}", flush=True); continue
    mae = float(np.abs(pre - test_hu).mean())
    corr = float(np.corrcoef(pre.ravel(), test_hu.ravel())[0, 1])
    print(f"  {pid}: coarse HU-MAE(test vs precomp) = {mae:6.1f} | corr = {corr:.3f} | "
          f"pre[min/mean/max]={pre.min():.0f}/{pre.mean():.0f}/{pre.max():.0f} "
          f"test={test_hu.min():.0f}/{test_hu.mean():.0f}/{test_hu.max():.0f}", flush=True)
print(">>> big HU-MAE / low corr => test coarse != training coarse => the sCT test was BUGGY (wrong coarse)", flush=True)
