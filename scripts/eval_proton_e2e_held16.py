"""Held16-vs-GT plan gamma for the NATIVE 1x1x3 proton-MRI E2E (runs/e2e_1x1x3_protonmri).
Answers: does the dose-aware 1x1x3 E2E beat the 2mm baseline (Proton-MRI 94.8 held16, in-sample)?
Pipeline per held16 pid: MR -> E2E.sct01 (native whole-image synth) -> density(=sct01*2.5) ->
predict_beams(net.dose) -> accumulate plan -> local gamma 1%/1mm vs GT plan (proton_ssd cache beamlets).
Usage: CUDA_VISIBLE_DEVICES=1 python scripts/eval_proton_e2e_held16.py [<ckpt.pt>]"""
import os, sys, json
from pathlib import Path
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, yaml, SimpleITK as sitk
from train_sct_paired import norm_mr, load_arr
from train_dose_e2e import E2E
from container.proton.predict import predict_beams
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array

CT_LO, CT_HI = -1000., 2000.; DENS_MAX = 2.5; DEV = "cuda"
# DOSERAD_E2E_CFG / DOSERAD_COARSE override the config + coarse dir for the WHOLE-image sCT variant
# (ref_..._whole2 / coarse_..._whole_soft); defaults keep the original 2mm behaviour.
CFG = yaml.safe_load(open(os.environ.get("DOSERAD_E2E_CFG", "configs/experiments/all75/e2e_1x1x3_protonmri.yaml")))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_protonmri/best.pt"
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
COARSE = os.environ.get("DOSERAD_COARSE", "/data/kwang/coarse_ct_1x1x3_samefield_soft")
GTCACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd")
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"]

net = E2E(CFG).to(DEV).eval()
sd = torch.load(CKPT, map_location="cpu"); net.load_state_dict(sd.get("ema", sd.get("model")))
_SYO = os.environ.get("DOSERAD_SYNTH_OVERRIDE")   # replace net.synth with a synth_ckpt (e.g. step-0 refiner)
if _SYO:
    ws = torch.load(_SYO, map_location="cpu"); wss = ws.get("ema", ws.get("model", ws))
    net.synth.load_state_dict({k[len("synth."):]: v for k, v in wss.items() if k.startswith("synth.")})
    print(f"[synth override] net.synth <- {_SYO}", flush=True)
pm = ProtonMachineData(device=DEV)
print(f"[eval] ckpt={CKPT}  step={sd.get('step','?')}", flush=True)

class _Img:
    def __init__(self, ct_sitk, dens):
        self.array = dens; self.spacing = ct_sitk.GetSpacing(); self.origin = ct_sitk.GetOrigin()
def to01(hu): return np.clip((hu - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)

_REALCT = os.environ.get("DOSERAD_REALCT") == "1"   # sanity: density from REAL CT (pipeline check)
_ANCH = None
@torch.no_grad()
def e2e_density(pid):
    if _REALCT:
        global _ANCH
        if _ANCH is None:
            import json as _j
            ent = _j.load(open(os.environ["DOSERAD_MACHINE"]))["hu_to_density"]["entries"]
            _ANCH = tuple(sorted((float(e["hu"]), float(e["density_g_cm3"])) for e in ent))
        from doserad.physics.density import hu_to_density
        hu = sitk.GetArrayFromImage(sitk.ReadImage(f"{PROT}/{pid}/image/ct.mha")).astype(np.float32)  # (z,y,x)
        return hu_to_density(hu, _ANCH).astype(np.float32)
    # MATCH the E2E training's load_vol EXACTLY: GetArrayFromImage (z,y,x), NOT load_arr (x,y,z).
    mr = norm_mr(sitk.GetArrayFromImage(sitk.ReadImage(f"{PROT}/{pid}/image/mr.mha")).astype(np.float32))  # (z,y,x)
    co = to01(sitk.GetArrayFromImage(sitk.ReadImage(f"{COARSE}/{pid}.nii.gz")).astype(np.float32))          # (z,y,x)
    inp = torch.from_numpy(np.stack([mr, co], 0)[None]).to(DEV)                      # (1,2,z,y,x)
    with torch.autocast("cuda"):
        sct01 = net.sct01(inp)[0, 0]                                                 # (z,y,x) native
    return (sct01.clamp(0, 1) * DENS_MAX).float().cpu().numpy()                      # (z,y,x) density (no transpose)

def beams_of(pid):
    plan = json.load(open(f"{PROT}/{pid}/{pid}.json")); beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for ri, r in enumerate(b["rays"]):
            r["ray_idx"] = ri
            for bl in r["beamlets"]: bl["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return beams

def gt_plan(pid, full):
    cps = []
    for f in sorted(GTCACHE.joinpath(pid).glob("B*_R*_L*.npz")):
        if ".tmp" in f.name: continue
        z = np.load(f); cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
    return accumulate_plan(cps, full)

rows = []
for pid in PIDS:
    ct = sitk.ReadImage(f"{PROT}/{pid}/image/ct.mha"); full = sitk.GetArrayFromImage(ct).shape
    dens = e2e_density(pid)
    preds = predict_beams(_Img(ct, dens), beams_of(pid), dens, torch.as_tensor(dens, device=DEV), net.dose, pm, DEV)
    pp = accumulate_plan([(d, bb) for d, bb in preds.values()], full)
    gt = gt_plan(pid, full); rx = float(gt.max())
    if rx <= 0: print(f"  {pid}: no GT, skip"); continue
    zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
    cr = (slice(max(int(zz.min())-m,0),int(zz.max())+m+1), slice(max(int(yy.min())-m,0),int(yy.max())+m+1), slice(max(int(xx.min())-m,0),int(xx.max())+m+1))
    g1c, g1m = gamma_array(pp[cr], gt[cr], ct.GetSpacing(), rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m] <= 1.0).mean()) * 100 if g1m.any() else float("nan")
    site = "lung" if "THB" in pid else "abd"
    rows.append(g1); print(f"  {pid} ({site}): gamma1/1 {g1:.1f}   [{len(preds)} beamlets]", flush=True)
print(f"\n>>> proton-MRI 1x1x3 E2E held16 gamma 1%/1mm = {np.nanmean(rows):.2f}   (2mm baseline held16 = 94.8, in-sample)")