"""Photon margin sweep on held16 via the EXACT container deploy path. For a given task + margin
(DOSERAD_PHOTON_MARGIN set in the shell BEFORE this runs — the app reads it at import), report:
  - official Beam MAE (per CP: region >=10% of that CP's max GT, |pred-gt| normalized by CP max)
  - plan gamma 1%/1mm  (⚠️ vs the margin-24 GT cache -> internal gamma OVER-penalizes small margin,
    like the OLD leaderboard; trust Beam MAE + timing more, treat gamma as a soft floor)
  - mean crop voxels + total predict wall-time (runtime proxy; T4 photon dose net is compute-bound)

Usage: DOSERAD_PHOTON_MARGIN=16 CUDA_VISIBLE_DEVICES=1 python scripts/margin_sweep.py photon_ct [N]
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)

TASK = sys.argv[1]
NPID = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MARGIN = os.environ.get("DOSERAD_PHOTON_MARGIN", "24")   # respect the shell; do NOT force 24
FROZEN = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))
PIDS = FROZEN["held16"][:NPID]
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
CLF = "/data/kwang/sct_classify_runs/clf_whole/best.pt"
MACH_PHOTON = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CFG = {
    "photon_ct": dict(weights=f"{RUNS}/docker_extracted/photon_ct_docker.pt", label="docker:p2",
                      cache="/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24"),
    "photon_mri": dict(weights=f"{RUNS}/docker_extracted/photon_mri_docker.pt", label="docker:scheme2-p4",
                       cache="/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24",
                       config="configs/experiments/all75/m24S2_p4_mmB.yaml"),
}[TASK]

os.environ["TORCHDYNAMO_DISABLE"] = "1"                  # sm_120 can't inductor-codegen; eager==compiled
os.environ.setdefault("DOSERAD_PHOTON_MARGIN", "24")    # keep whatever the shell set
os.environ["DOSERAD_WEIGHTS"] = os.environ.get("DOSERAD_W_OVERRIDE", CFG["weights"])
os.environ["DOSERAD_MACHINE"] = MACH_PHOTON
if "config" in CFG:
    os.environ["DOSERAD_CONFIG"] = CFG["config"]; os.environ["DOSERAD_CLF"] = CLF

import numpy as np, SimpleITK as sitk
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
if TASK == "photon_ct":   from container.photon import app
elif TASK == "photon_mri": from container.photon_mri import app
else: raise SystemExit(f"unknown/unsupported task {TASK}")
CACHE = Path(CFG["cache"])

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for cp in b["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return {"image_file_idx": 0, "beams": beams}

def load_src(pid):
    if TASK == "photon_ct": im = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/ct.mha")
    else:                   im = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/mr.mha")
    return im

def _embed(crop, bb, ush, off):
    a = np.zeros(ush, np.float32); a[off[0]:off[0]+crop.shape[0], off[1]:off[1]+crop.shape[1], off[2]:off[2]+crop.shape[2]] = crop; return a

def score(pid):
    src = load_src(pid); plan = json.load(open(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    t0 = time.time(); preds = app._predict_fn(src, build_entry(plan)); dt = time.time() - t0
    full = sitk.GetArrayFromImage(src).shape
    pred_cps, gt_cps, beam_mae, vox = [], [], [], []
    for f in sorted((CACHE / pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        z = np.load(f); gbb = tuple(int(v) for v in z["bbox"]); gt = z["dose"].astype(np.float32)
        gt_cps.append((gt, gbb))
        if (bi, cpi) not in preds: continue
        pcrop, pbb, _ = preds[(bi, cpi)]; pred_cps.append((pcrop, pbb)); vox.append(int(pcrop.size))
        gm = float(gt.max())
        if gm <= 0: continue
        uz0=min(gbb[0],pbb[0]); uz1=max(gbb[1],pbb[1]); uy0=min(gbb[2],pbb[2]); uy1=max(gbb[3],pbb[3]); ux0=min(gbb[4],pbb[4]); ux1=max(gbb[5],pbb[5])
        ush=(uz1-uz0+1, uy1-uy0+1, ux1-ux0+1)
        G=_embed(gt,gbb,ush,(gbb[0]-uz0,gbb[2]-uy0,gbb[4]-ux0)); P=_embed(pcrop,pbb,ush,(pbb[0]-uz0,pbb[2]-uy0,pbb[4]-ux0))
        hi = G >= 0.10*gm
        if hi.sum(): beam_mae.append(float(np.abs(P-G)[hi].mean()/gm))
    pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full); rx = float(gt.max())
    zz,yy,xx = np.where(gt >= 0.05*rx); m=4
    cr=(slice(max(int(zz.min())-m,0),int(zz.max())+m+1),slice(max(int(yy.min())-m,0),int(yy.max())+m+1),slice(max(int(xx.min())-m,0),int(xx.max())+m+1))
    g1c,g1m = gamma_array(pp[cr], gt[cr], src.GetSpacing(), rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m]<=1.0).mean())*100 if g1m.any() else float("nan")
    return g1, float(np.mean(beam_mae)), float(np.mean(vox)), dt, len(pred_cps)

app.load_models()
print(f"[margin_sweep] task={TASK} MARGIN={MARGIN} weights={CFG['label']} N={len(PIDS)}", flush=True)
G, BM, VOX, DT = [], [], [], []
for pid in PIDS:
    site = "lung" if "THB" in pid else "abd"
    g1, bm, vox, dt, npd = score(pid)
    G.append(g1); BM.append(bm); VOX.append(vox); DT.append(dt)
    print(f"  {pid} ({site}): gamma {g1:5.1f}  BeamMAE {bm:.4f}  vox {vox/1e6:5.2f}M  {dt:4.0f}s  [{npd}cp]", flush=True)
print(f"\n>>> {TASK} MARGIN={MARGIN}: gamma {np.nanmean(G):.2f} | BeamMAE {np.nanmean(BM):.4f} | "
      f"mean crop {np.mean(VOX)/1e6:.2f}M vox | predict {np.mean(DT):.0f}s/pt", flush=True)
