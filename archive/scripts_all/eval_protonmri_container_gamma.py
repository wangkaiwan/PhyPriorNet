"""ACCEPTANCE test: held16 gamma using the CONTAINER path (live clf_1x1x3_samefield -> coarse ->
E2E synth -> density, native grid), vs the eval's precomputed-coarse 93.04. If container gamma ~= 93,
the small live-vs-precomputed coarse Δ is immaterial and the container is faithful -> safe to ship.
Self-contained (does NOT import eval_proton_e2e_held16, whose module body runs a full eval on import).

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/eval_protonmri_container_gamma.py <e2e_ckpt.pt> [clf.pt]
"""
import os, sys, json
from pathlib import Path
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, yaml, SimpleITK as sitk
from train_dose_e2e import E2E
from container.mri_synth import synth_density, load_classifier
from container.proton.predict import predict_beams
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_protonmri/ckpt_30k.pt"
CLF  = sys.argv[2] if len(sys.argv) > 2 else "/data/kwang/sct_classify_runs/clf_1x1x3_samefield/best.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = "configs/experiments/all75/e2e_1x1x3_protonmri.yaml"
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
GTCACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd")
cfg = yaml.safe_load(open(CFG))

class _Img:
    def __init__(self, ct_sitk, dens):
        self.array = dens; self.spacing = ct_sitk.GetSpacing(); self.origin = ct_sitk.GetOrigin()

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

net = E2E(cfg).to(DEV).eval()
sd = torch.load(CKPT, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
clf = load_classifier(CLF, DEV)
pm = ProtonMachineData(device=DEV)
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"]
print(f"[container-gamma] ckpt={CKPT} step={sd.get('step','?')} clf={CLF}", flush=True)

rows = []
for pid in PIDS:
    ct = sitk.ReadImage(f"{PROT}/{pid}/image/ct.mha"); full = sitk.GetArrayFromImage(ct).shape
    mr_sitk = sitk.ReadImage(f"{PROT}/{pid}/image/mr.mha")
    dens, _ = synth_density(mr_sitk, clf, net, DEV, density_direct=True, native_grid=True)   # container path
    preds = predict_beams(_Img(ct, dens), beams_of(pid), dens, torch.as_tensor(dens, device=DEV), net.dose, pm, DEV)
    pp = accumulate_plan([(d, bb) for d, bb in preds.values()], full)
    gt = gt_plan(pid, full); rx = float(gt.max())
    if rx <= 0: continue
    zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
    cr = (slice(max(int(zz.min())-m,0),int(zz.max())+m+1), slice(max(int(yy.min())-m,0),int(yy.max())+m+1), slice(max(int(xx.min())-m,0),int(xx.max())+m+1))
    g1c, g1m = gamma_array(pp[cr], gt[cr], ct.GetSpacing(), rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m] <= 1.0).mean()) * 100 if g1m.any() else float("nan")
    site = "lung" if "THB" in pid else "abd"
    rows.append(g1); print(f"  {pid} ({site}): gamma1/1 {g1:.1f}", flush=True)
print(f"\n>>> CONTAINER-path proton-MRI held16 gamma = {np.nanmean(rows):.2f}   (eval precomputed-coarse = 93.04 @30k)")
