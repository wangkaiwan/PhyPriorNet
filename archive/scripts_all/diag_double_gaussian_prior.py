"""Validate the proton double-Gaussian (nuclear-halo) prior BEFORE any retrain. The residual net learns
GT - PB_prior, so a prior CLOSER to the GT MC dose = smaller residual = easier to learn = likely better
gamma. This compares the PRIOR-ONLY plan (no net; the PB prior is ~95% of the proton dose) to the GT plan
via local 1%/1mm gamma, SINGLE-Gaussian vs DOUBLE-Gaussian (build_ray with DOSERAD_LATERAL_DOUBLE=1), on
fold-0 val proton patients. If double >> single, a retrain of the residual net on the double prior is
justified; if similar/worse, drop the double-Gaussian idea (only build_ray touched -> trivial revert).

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/diag_double_gaussian_prior.py [N]
"""
import os, sys, json
from pathlib import Path
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DOSERAD_2PASS"] = "0"          # plain build_ray so the lateral toggle is exercised directly
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
import numpy as np, torch, SimpleITK as sitk
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.physics.density import hu_to_density
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from accel.proton_build_ray import build_ray
from container.proton.geom_bbox import geom_bbox_proton
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array

DEV = "cuda"
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
GTCACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"][:N]
machine = ProtonMachineData(device=DEV)
_ent = json.load(open(os.environ["DOSERAD_MACHINE"]))["hu_to_density"]["entries"]
ANCH = tuple(sorted((float(e["hu"]), float(e["density_g_cm3"])) for e in _ent))
_SCALE2 = float(_P_CH_SCALE_PRIOR[2])

class _Img:
    def __init__(self, ct, dens): self.array = dens; self.spacing = ct.GetSpacing(); self.origin = ct.GetOrigin()

def beams_of(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for ri, r in enumerate(b["rays"]):
            r["ray_idx"] = ri
        beams.append(b)
    return beams

def gt_plan(pid, full):
    cps = []
    for f in sorted(GTCACHE.joinpath(pid).glob("B*_R*_L*.npz")):
        if ".tmp" in f.name: continue
        z = np.load(f); cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
    return accumulate_plan(cps, full)

@torch.no_grad()
def prior_plan(double, image, dens_t, beams, full):
    os.environ["DOSERAD_LATERAL_DOUBLE"] = "1" if double else "0"
    cps = []
    for b in beams:
        for r in b["rays"]:
            bls = []
            for bl in r["beamlets"]:
                gb = geom_bbox_proton(image.array, image.spacing, image.origin,
                                      r["ray_source"], r["ray_target"], machine, bl["energy"])
                if gb is None: continue
                bls.append(dict(energy=bl["energy"], bbox=gb))
            if not bls: continue
            stacks = build_ray(image, r["ray_source"], r["ray_target"], bls,
                               machine=machine, density=dens_t, device=DEV)
            for (stack, gbb), bl in zip(stacks, bls):
                pb = (stack[2] * _SCALE2 / PROTON_DOSE_SCALE).float().cpu().numpy()
                cps.append((pb, gbb))
    return accumulate_plan(cps, full)

print(f"[double-g] N={len(PIDS)}  (prior-only plan gamma vs GT, single vs double)", flush=True)
S, D = [], []
for pid in PIDS:
    ct = sitk.ReadImage(f"{PROT}/{pid}/image/ct.mha"); full = sitk.GetArrayFromImage(ct).shape
    hu = sitk.GetArrayFromImage(ct).astype(np.float32)
    dens = hu_to_density(hu, ANCH).astype(np.float32); dens_t = torch.as_tensor(dens, device=DEV)
    image = _Img(ct, dens)
    plan = json.load(open(f"{PROT}/{pid}/{pid}.json")); beams = beams_of(plan)
    gt = gt_plan(pid, full); rx = float(gt.max())
    if rx <= 0: print(f"  {pid}: no GT, skip"); continue
    ps = prior_plan(False, image, dens_t, beams, full)
    pd = prior_plan(True, image, dens_t, beams, full)
    zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
    cr = (slice(max(int(zz.min())-m,0), int(zz.max())+m+1), slice(max(int(yy.min())-m,0), int(yy.max())+m+1),
          slice(max(int(xx.min())-m,0), int(xx.max())+m+1))
    gsc, gsm = gamma_array(ps[cr], gt[cr], ct.GetSpacing(), rx, dose_pct=1.0, dta_mm=1.0)
    gdc, gdm = gamma_array(pd[cr], gt[cr], ct.GetSpacing(), rx, dose_pct=1.0, dta_mm=1.0)
    gs = float((gsc[gsm] <= 1).mean()) * 100 if gsm.any() else float("nan")
    gd = float((gdc[gdm] <= 1).mean()) * 100 if gdm.any() else float("nan")
    site = "lung" if "THB" in pid else "abd"
    S.append(gs); D.append(gd)
    print(f"  {pid} ({site}): single-prior {gs:5.1f} | double-prior {gd:5.1f} | delta {gd-gs:+.1f}", flush=True)
print(f"\n>>> PRIOR-ONLY plan gamma: single {np.nanmean(S):.2f} | double {np.nanmean(D):.2f} | "
      f"delta {np.nanmean(D)-np.nanmean(S):+.2f}  (>0 => double prior closer to GT => retrain justified)", flush=True)
