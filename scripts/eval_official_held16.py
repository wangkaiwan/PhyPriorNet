"""Held16 eval with the OFFICIAL metrics (vendored scripts/official_eval/) — to diagnose the IDD
collapse + calibrate margin under the real challenge口径. For a task + margin, report per-CP IDD
(propagation-direction-aware) + official Beam MAE + region-Rx Stratified (thoracic 70Gy / abdominal
60Gy) + plan gamma (our pymedphys == official). Env DOSERAD_PHOTON_MARGIN controls the crop margin.

IDD note: official computes the depth-dose on the FULL transverse plane (rotation pivots on the grid
centre). We z-collapse each CP crop, embed the 2D plane into the full (ny,nx) grid at its bbox, then
call the official compute_idd_curve via a (1,ny,nx) array — identical geometry, 2D cost.

Usage: DOSERAD_PHOTON_MARGIN=24 CUDA_VISIBLE_DEVICES=1 python scripts/eval_official_held16.py photon_ct [N]
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
TASK = sys.argv[1]
NPID = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MARGIN = os.environ.get("DOSERAD_PHOTON_MARGIN", "24")
FROZEN = json.load(open((os.environ.get("WORKDIR", "./workdir") + "/eval_cohort_frozen.json")))
PIDS = FROZEN["held16"][:NPID]
RUNS = (os.environ.get("WORKDIR", "./workdir") + "/runs")
CLF = (os.environ.get("WORKDIR", "./workdir") + "/sct_runs" + "/clf_whole/best.pt")
MACH_PHOTON = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/photon/training/beam_parameters.json")
PHOTON_ROOT = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/photon/training")
CFG = {
    "photon_ct": dict(weights=f"{RUNS}/docker_extracted/photon_ct_docker.pt", label="docker:p2",
                      cache=f(os.environ.get("WORKDIR", "./workdir") + "/cache/crops/photon_skinentry_m24")),
    "photon_mri": dict(weights=f"{RUNS}/docker_extracted/photon_mri_docker.pt", label="docker:scheme2-p4",
                       cache=f(os.environ.get("WORKDIR", "./workdir") + "/cache/crops/photon_skinentry_m24"),
                       config="configs/experiments/all75/m24S2_p4_mmB.yaml"),
}[TASK]
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("DOSERAD_PHOTON_MARGIN", "24")
os.environ["DOSERAD_WEIGHTS"] = os.environ.get("DOSERAD_W_OVERRIDE", CFG["weights"])
os.environ["DOSERAD_MACHINE"] = MACH_PHOTON
if "config" in CFG:
    os.environ["DOSERAD_CONFIG"] = os.environ.get("DOSERAD_CFG_OVERRIDE", CFG["config"])
    os.environ["DOSERAD_CLF"] = os.environ.get("DOSERAD_CLF_OVERRIDE", CLF)

import numpy as np, SimpleITK as sitk
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
from official_eval.metrics_beam import masked_beam_mae, idd_curve_distance, directions_of
from official_eval.metrics_plan import stratified_plan_mae
if TASK == "photon_ct":   from container.photon import app
elif TASK == "photon_mri": from container.photon_mri import app
else: raise SystemExit(f"unsupported task {TASK}")
CACHE = Path(CFG["cache"])

def rx_of(pid):   # region prescription: thoracic 1TH*=70, abdominal 1AB*=60
    return 70.0 if pid.upper().startswith(("1TH", "AUMC")) else 60.0

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for cp in b["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return {"image_file_idx": 0, "beams": beams}

def plane2d(crop, bbox, full_yx):
    """z-collapse a (z,y,x) crop, embed into the full (ny,nx) plane at its bbox -> (1,ny,nx)."""
    z0, z1, y0, y1, x0, x1 = bbox
    p = np.zeros(full_yx, np.float64)
    p[y0:y1+1, x0:x1+1] = crop.sum(axis=0, dtype=np.float64)
    return p[None]

def score(pid):
    src = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/{'ct' if TASK=='photon_ct' else 'mr'}.mha")
    plan = json.load(open(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    preds = app._predict_fn(src, build_entry(plan))
    full = sitk.GetArrayFromImage(src).shape            # (z,y,x)
    ny, nx = full[1], full[2]
    dirs = directions_of({"beams": [dict(b, beam_idx=bi) for bi, b in enumerate(plan["beams"])]})
    sp = np.array(src.GetSpacing(), float)              # (x,y,z)
    idd, bmae, pred_cps, gt_cps = [], [], [], []
    for f in sorted((CACHE / pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        z = np.load(f); gbb = tuple(int(v) for v in z["bbox"]); gt = z["dose"].astype(np.float32)
        gt_cps.append((gt, gbb))
        if (bi, cpi) not in preds: continue
        pcrop, pbb, _ = preds[(bi, cpi)]; pred_cps.append((pcrop, pbb))
        # official Beam MAE (align pred+gt on union bbox)
        uz0=min(gbb[0],pbb[0]);uz1=max(gbb[1],pbb[1]);uy0=min(gbb[2],pbb[2]);uy1=max(gbb[3],pbb[3]);ux0=min(gbb[4],pbb[4]);ux1=max(gbb[5],pbb[5])
        ush=(uz1-uz0+1,uy1-uy0+1,ux1-ux0+1)
        G=np.zeros(ush,np.float32); G[gbb[0]-uz0:gbb[1]-uz0+1,gbb[2]-uy0:gbb[3]-uy0+1,gbb[4]-ux0:gbb[5]-ux0+1]=gt
        P=np.zeros(ush,np.float32); P[pbb[0]-uz0:pbb[1]-uz0+1,pbb[2]-uy0:pbb[3]-uy0+1,pbb[4]-ux0:pbb[5]-ux0+1]=pcrop
        bmae.append(masked_beam_mae(P, G))
        d = dirs.get((bi, cpi))
        if d is not None and abs(d[2]) < 1e-9:
            idd.append(idd_curve_distance(plane2d(pcrop,pbb,(ny,nx)), plane2d(gt,gbb,(ny,nx)), d, sp))
    pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full)
    rx = rx_of(pid)
    strat = stratified_plan_mae(pp, gt, prescription_dose=rx)
    gmax = float(gt.max()); zz,yy,xx = np.where(gt>=0.05*gmax); m=4
    cr=(slice(max(int(zz.min())-m,0),int(zz.max())+m+1),slice(max(int(yy.min())-m,0),int(yy.max())+m+1),slice(max(int(xx.min())-m,0),int(xx.max())+m+1))
    g1c,g1m = gamma_array(pp[cr], gt[cr], src.GetSpacing(), gmax, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m]<=1.0).mean())*100 if g1m.any() else float("nan")
    return np.nanmean(idd), np.nanmean(bmae), strat["mae_stratified"], g1, rx

app.load_models()
print(f"[official-eval] task={TASK} MARGIN={MARGIN} weights={CFG['label']} N={len(PIDS)}", flush=True)
IDD,BM,ST,G = [],[],[],[]
for pid in PIDS:
    site="lung/thor" if pid.upper().startswith("1TH") else "abd"
    idd,bm,st,g,rx = score(pid); IDD.append(idd);BM.append(bm);ST.append(st);G.append(g)
    print(f"  {pid} ({site},Rx{int(rx)}): IDD {idd:.4f}  BeamMAE {bm:.4f}  Strat {st:.4f}  gamma {g:.1f}", flush=True)
print(f"\n>>> {TASK} MARGIN={MARGIN}: IDD {np.nanmean(IDD):.4f} | BeamMAE {np.nanmean(BM):.4f} | "
      f"Stratified {np.nanmean(ST):.4f} | gamma {np.nanmean(G):.2f}", flush=True)
