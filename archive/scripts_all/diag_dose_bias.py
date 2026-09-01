"""Systematic dose-calibration bias check on held16 (photon-CT) — the last #1 lever.

DVH (D98/V95/D2/Dmean), Beam MAE and Stratified MAE are all |pred-gt| dose errors normalized by Rx;
a GLOBAL multiplicative bias b (pred ~= b * gt) contributes |b-1| directly to all of them. If a
consistent b != 1 exists, a single output gain 1/b improves DVH+Beam+Strat at ZERO accuracy risk.
Measures, per patient: b_high = sum(pred*gt)/sum(gt^2) over the high-dose region (gt >= 50% max,
~PTV proxy = what D98/V95 sees) and b_mid (10-50%, OAR-ish), plus what the gain correction would do
to the plan MAE. Uses the SAME deploy path as the board entry (m16, deployed weights).

Usage: DOSERAD_PHOTON_MARGIN=16 CUDA_VISIBLE_DEVICES=1 python scripts/diag_dose_bias.py [N]
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("DOSERAD_PHOTON_MARGIN", "16")
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
os.environ["DOSERAD_WEIGHTS"] = os.environ.get("DOSERAD_W_OVERRIDE", f"{RUNS}/docker_extracted/photon_ct_docker.pt")
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
os.environ["DOSERAD_MODALITY"] = "ct"
import numpy as np, SimpleITK as sitk
from doserad.eval.plan_agg import accumulate_plan
from container.photon import app

NPID = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FROZEN = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))
PIDS = FROZEN["held16"][:NPID]
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        bb = dict(b); bb["beam_idx"] = bi
        for cp in bb["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(bb)
    return {"image_file_idx": 0, "beams": beams}

app.load_models()
rows = []
print(f"[bias] N={len(PIDS)} margin={os.environ['DOSERAD_PHOTON_MARGIN']}", flush=True)
for pid in PIDS:
    src = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/ct.mha")
    plan = json.load(open(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    preds = app._predict_fn(src, build_entry(plan))
    full = sitk.GetArrayFromImage(src).shape
    pred_cps, gt_cps = [], []
    for f in sorted((CACHE / pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        z = np.load(f); gt_cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
        if (bi, cpi) in preds:
            pcrop, pbb, _ = preds[(bi, cpi)]; pred_cps.append((pcrop, pbb))
    pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full)
    gmax = float(gt.max())
    hi = gt >= 0.5 * gmax; mid = (gt >= 0.1 * gmax) & ~hi
    b_hi  = float((pp[hi] * gt[hi]).sum() / (gt[hi] ** 2).sum())
    b_mid = float((pp[mid] * gt[mid]).sum() / (gt[mid] ** 2).sum())
    # MAE in the scored region before/after a global gain 1/b_hi
    m = gt >= 0.1 * gmax
    mae0 = float(np.abs(pp[m] - gt[m]).mean() / gmax)
    mae1 = float(np.abs(pp[m] / b_hi - gt[m]).mean() / gmax)
    rows.append((pid, b_hi, b_mid, mae0, mae1))
    print(f"  {pid}: b_high {b_hi:.4f}  b_mid {b_mid:.4f}  MAE {mae0*100:.3f}% -> gain-corrected {mae1*100:.3f}%", flush=True)

bh = np.array([r[1] for r in rows]); bm = np.array([r[2] for r in rows])
m0 = np.array([r[3] for r in rows]); m1 = np.array([r[4] for r in rows])
print(f"\n>>> b_high mean {bh.mean():.4f} +- {bh.std():.4f} (range {bh.min():.4f}..{bh.max():.4f})")
print(f">>> b_mid  mean {bm.mean():.4f} +- {bm.std():.4f}")
print(f">>> MAE {m0.mean()*100:.3f}% -> per-patient-gain {m1.mean()*100:.3f}%")
print(">>> READ: |mean(b_high)-1| >~ 0.005 AND consistent sign (std < |bias|) => a global gain is a real")
print(">>> DVH/Beam/Strat lever. If b straddles 1 patient-to-patient, a global gain won't help (no lever).")
