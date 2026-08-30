"""Decompose the DEPLOYED photon-MRI (scheme2-p4, md5 7e05dbdc) per-CP dose error by dose region on
held16, to settle WHY our Beam MAE (rank 4) + Stratified MAE (rank 5) are weak.
Official口径 (metrics-and-ranking):
  - Beam MAE: per beam(CP), region = voxels >= 10% of THAT beam's max GT; MAE normalized by beam max.
  - Stratified Plan MAE: 3 strata of prescription — high >=80%, mid 30-80%, low 10-30% (we proxy Rx=plan max).
Hypotheses under test: (mine) Beam-MAE error concentrates in the high-dose BULK (grad/het/lung over-weight
edges); (user's) the LOW stratum (10-30%) is under-fit (hi_w=10 starves <10% beam max)."""
import os, sys, json
from pathlib import Path
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DOSERAD_PHOTON_MARGIN"] = "24"
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
os.environ["DOSERAD_WEIGHTS"] = os.environ.get("DOSERAD_W_OVERRIDE", f"{RUNS}/docker_extracted/photon_mri_docker.pt")
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
os.environ["DOSERAD_CONFIG"] = "configs/experiments/all75/m24S2_p4_mmB.yaml"
os.environ["DOSERAD_CLF"] = "/data/kwang/sct_classify_runs/clf_whole/best.pt"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, SimpleITK as sitk
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
from container.photon_mri import app

PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"][:int(os.environ.get("DOSERAD_NEVAL","16"))]

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for cp in b["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return {"image_file_idx": 0, "beams": beams}

def embed(crop, bb, shape):
    z0, z1, y0, y1, x0, x1 = bb
    a = np.zeros(shape, np.float32); a[z0:z1+1, y0:y1+1, x0:x1+1] = crop; return a

# per-CP Beam MAE (official) + within-region dose-bin decomposition
BINS = [(0.10, 0.30), (0.30, 0.60), (0.60, 0.90), (0.90, 1.01)]   # of CP max, within the >=10% Beam-MAE region
app.load_models()
beam_mae, below10_frac, gammas = [], [], []
bin_abserr = np.zeros(len(BINS)); bin_vox = np.zeros(len(BINS))
strat = {"high(>=80%)": [], "mid(30-80%)": [], "low(10-30%)": []}
for pid in PIDS:
    mr = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/mr.mha")
    plan = json.load(open(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    preds = app._predict_fn(mr, build_entry(plan))
    full = sitk.GetArrayFromImage(mr).shape
    # --- per-CP Beam MAE ---
    pred_cps, gt_cps = [], []
    for f in sorted(CACHE.joinpath(pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        z = np.load(f); gbb = tuple(int(v) for v in z["bbox"]); gt = z["dose"].astype(np.float32)
        gt_cps.append((gt, gbb))
        if (bi, cpi) not in preds: continue
        pcrop, pbb, _ = preds[(bi, cpi)]
        gm = float(gt.max())
        if gm <= 0: continue
        uz0=min(gbb[0],pbb[0]); uz1=max(gbb[1],pbb[1]); uy0=min(gbb[2],pbb[2]); uy1=max(gbb[3],pbb[3]); ux0=min(gbb[4],pbb[4]); ux1=max(gbb[5],pbb[5])
        ush=(uz1-uz0+1, uy1-uy0+1, ux1-ux0+1)
        G=embed(gt,(gbb[0]-uz0,gbb[1]-uz0,gbb[2]-uy0,gbb[3]-uy0,gbb[4]-ux0,gbb[5]-ux0),ush)
        P=embed(pcrop,(pbb[0]-uz0,pbb[1]-uz0,pbb[2]-uy0,pbb[3]-uy0,pbb[4]-ux0,pbb[5]-ux0),ush)
        ae=np.abs(P-G); hi=G>=0.10*gm
        if hi.sum(): beam_mae.append(ae[hi].mean()/gm)                       # official Beam MAE (norm by beam max)
        below=(G>0)&(G<0.10*gm)
        below10_frac.append(ae[below].sum()/max(ae[G>0].sum(),1e-9))         # how much error sits <10% (EXCLUDED)
        for k,(lo,hi2) in enumerate(BINS):                                    # within-region decomposition
            m=(G>=lo*gm)&(G<hi2*gm)
            if m.sum(): bin_abserr[k]+=ae[m].sum()/gm; bin_vox[k]+=m.sum()
    pred_cps=[(preds[k][0],preds[k][1]) for f in sorted(CACHE.joinpath(pid).glob("*.npz")) if ".tmp" not in f.name for k in [tuple(int(x) for x in f.stem.split("_"))] if k in preds]
    gt_cps=[(np.load(f)["dose"].astype(np.float32), tuple(int(v) for v in np.load(f)["bbox"])) for f in sorted(CACHE.joinpath(pid).glob("*.npz")) if ".tmp" not in f.name]
    pp=accumulate_plan(pred_cps,full); gt=accumulate_plan(gt_cps,full); rx=float(gt.max())
    for nm,(lo,hi2) in [("high(>=80%)",(0.80,1.01)),("mid(30-80%)",(0.30,0.80)),("low(10-30%)",(0.10,0.30))]:
        m=(gt>=lo*rx)&(gt<hi2*rx)
        if m.sum(): strat[nm].append(np.abs(pp-gt)[m].mean()/rx)
    zz,yy,xx=np.where(gt>=0.05*rx); mg=4
    cr=(slice(max(int(zz.min())-mg,0),int(zz.max())+mg+1),slice(max(int(yy.min())-mg,0),int(yy.max())+mg+1),slice(max(int(xx.min())-mg,0),int(xx.max())+mg+1))
    g1c,g1m=gamma_array(pp[cr],gt[cr],mr.GetSpacing(),rx,dose_pct=1.0,dta_mm=1.0)
    gammas.append(float((g1c[g1m]<=1.0).mean())*100 if g1m.any() else float("nan"))
    print(f"  {pid}: BeamMAE~{np.mean(beam_mae):.4f} gamma~{np.nanmean(gammas):.1f}", flush=True)

print("\n=== photon-MRI error decomposition (held16) ===")
print(f"WEIGHTS: {os.environ['DOSERAD_WEIGHTS']}")
print(f">>> Beam MAE {np.mean(beam_mae):.4f}  |  Local Gamma 1%/1mm {np.nanmean(gammas):.2f}  (A0 deployed = 0.0100 / gamma baseline)")
print(f"Official Beam MAE (mean over CPs, norm by beam max): {np.mean(beam_mae):.4f}  (leaderboard test = 0.0170)")
print(f"Error mass BELOW 10% beam-max (EXCLUDED from Beam MAE): {np.mean(below10_frac)*100:.1f}% of each CP's total abs-error")
print("\nWithin the >=10% Beam-MAE region — where the error sits (abs-err share / voxel share):")
tot=bin_abserr.sum()
for k,(lo,hi2) in enumerate(BINS):
    print(f"  {int(lo*100):3d}-{int(hi2*100):3d}% beam-max:  err {bin_abserr[k]/tot*100:4.1f}%   vox {bin_vox[k]/bin_vox.sum()*100:4.1f}%   -> err/vox ratio {(bin_abserr[k]/tot)/(bin_vox[k]/bin_vox.sum()+1e-9):.2f}")
print("\nStratified Plan MAE by stratum (norm by plan-max proxy for Rx):")
for nm in strat:
    print(f"  {nm:12s}: {np.mean(strat[nm]):.4f}")
