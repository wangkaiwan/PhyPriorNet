"""Test the IDD-collapse hypothesis: is our leaderboard IDD (0.0189) high because our margin-24 PRED
crops lateral scatter that the FULL-GRID GT has? Compute per-CP IDD two ways for a few held16 patients:
  (A) pred(margin) vs GT(margin-24 cache)   -- what the internal eval does (got 0.0048)
  (B) pred(margin) vs GT(FULL-GRID dose/Dose_B*_CP*.mha)  -- closer to official (full-grid GT)
If (B) >> (A), the crop is the IDD culprit -> IDD wants a LARGER margin (conflicts with runtime).
Usage: DOSERAD_PHOTON_MARGIN=24 CUDA_VISIBLE_DEVICES=1 python scripts/diag_idd_fullgt.py [N]
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("DOSERAD_PHOTON_MARGIN", "24")
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
os.environ["DOSERAD_WEIGHTS"] = f"{RUNS}/docker_extracted/photon_ct_docker.pt"
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))["held16"]
PIDS = PIDS[:4] + PIDS[8:8+max(0, N-4)]     # a couple abd + a couple lung

import numpy as np, SimpleITK as sitk
from official_eval.metrics_beam import idd_curve_distance, directions_of
from container.photon import app

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for cp in b["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return {"image_file_idx": 0, "beams": beams}

def embed_full(crop, bbox, full):           # (z,y,x) crop -> full 3D
    z0,z1,y0,y1,x0,x1 = bbox
    a = np.zeros(full, np.float64); a[z0:z1+1,y0:y1+1,x0:x1+1] = crop; return a

app.load_models()
print(f"[idd-fullgt] MARGIN={os.environ['DOSERAD_PHOTON_MARGIN']}  N={len(PIDS)}", flush=True)
for pid in PIDS:
    src = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/ct.mha"); full = sitk.GetArrayFromImage(src).shape
    sp = np.array(src.GetSpacing(), float)
    plan = json.load(open(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    preds = app._predict_fn(src, build_entry(plan))
    dirs = directions_of({"beams": [dict(b, beam_idx=bi) for bi, b in enumerate(plan["beams"])]})
    A, B = [], []
    for f in sorted((CACHE / pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        if (bi, cpi) not in preds: continue
        d = dirs.get((bi, cpi))
        if d is None or abs(d[2]) >= 1e-9: continue
        z = np.load(f); gbb = tuple(int(v) for v in z["bbox"]); gt_c = z["dose"].astype(np.float64)
        pcrop, pbb, _ = preds[(bi, cpi)]
        pred_full = embed_full(pcrop.astype(np.float64), pbb, full)
        gt_cache_full = embed_full(gt_c, gbb, full)
        # FULL-GRID GT from the raw per-CP dose file
        gf = f"{PHOTON_ROOT}/{pid}/dose/Dose_B{bi}_CP{cpi:03d}.mha"
        if not os.path.exists(gf): continue
        gt_raw = sitk.GetArrayFromImage(sitk.ReadImage(gf)).astype(np.float64)
        A.append(idd_curve_distance(pred_full, gt_cache_full, d, sp))
        B.append(idd_curve_distance(pred_full, gt_raw,        d, sp))
    site = "lung" if "THB" in pid else "abd"
    print(f"  {pid} ({site}): IDD(A: vs m24-cache) {np.nanmean(A):.4f}   IDD(B: vs FULL-grid GT) {np.nanmean(B):.4f}   ratio {np.nanmean(B)/max(np.nanmean(A),1e-9):.1f}x", flush=True)
