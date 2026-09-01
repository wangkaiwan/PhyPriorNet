"""REDO the IDD-vs-margin test CORRECTLY: same model (deployed 4018f597), margins {8,16,24}, and apply
a minimum_cutoff to BOTH pred and full-grid GT before computing the official IDD. The official GT is a
cutoff-adhering version (low-dose scatter zeroed); our earlier test used the RAW GT (95% low-dose noise)
and wrongly concluded 'larger margin better'. We don't have the exact per-beam cutoff, so SWEEP it as a
fraction of each CP's GT max: C in {0, 0.005, 0.01, 0.02}. If applying a cutoff flips the trend so that
SMALLER margin gives <= IDD, that confirms the user's leaderboard observation (margin-8 IDD better).

Usage: DOSERAD_PHOTON_MARGIN=8 CUDA_VISIBLE_DEVICES=1 python scripts/diag_idd_cutoff.py [N]
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("DOSERAD_PHOTON_MARGIN", "24")
MARGIN = os.environ["DOSERAD_PHOTON_MARGIN"]
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
os.environ["DOSERAD_WEIGHTS"] = f"{RUNS}/docker_extracted/photon_ct_docker.pt"
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))["held16"]
PIDS = PIDS[:2] + PIDS[8:8 + max(0, N - 2)]     # 2 abd + (N-2) lung
CUTOFFS = [0.0, 0.005, 0.01, 0.02]              # fraction of each CP's GT max

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

def embed(crop, bbox, full):
    z0,z1,y0,y1,x0,x1 = bbox
    a = np.zeros(full, np.float64); a[z0:z1+1,y0:y1+1,x0:x1+1] = crop; return a

app.load_models()
print(f"[idd-cutoff] MARGIN={MARGIN}  N={len(PIDS)}  cutoffs(%GTmax)={[c*100 for c in CUTOFFS]}", flush=True)
agg = {c: [] for c in CUTOFFS}
for pid in PIDS:
    src = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/ct.mha"); full = sitk.GetArrayFromImage(src).shape
    sp = np.array(src.GetSpacing(), float)
    plan = json.load(open(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    preds = app._predict_fn(src, build_entry(plan))
    dirs = directions_of({"beams": [dict(b, beam_idx=bi) for bi, b in enumerate(plan["beams"])]})
    percp = {c: [] for c in CUTOFFS}
    for f in sorted((CACHE / pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        if (bi, cpi) not in preds: continue
        d = dirs.get((bi, cpi))
        if d is None or abs(d[2]) >= 1e-9: continue
        pcrop, pbb, _ = preds[(bi, cpi)]
        gf = f"{PHOTON_ROOT}/{pid}/dose/Dose_B{bi}_CP{cpi:03d}.mha"
        if not os.path.exists(gf): continue
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gf)).astype(np.float64)          # full-grid raw GT
        pr = embed(pcrop.astype(np.float64), pbb, full)                             # pred at margin M
        gmax = float(gt.max())
        if gmax <= 0: continue
        for c in CUTOFFS:
            thr = c * gmax
            g2 = np.where(gt < thr, 0.0, gt); p2 = np.where(pr < thr, 0.0, pr)      # cutoff BOTH
            percp[c].append(idd_curve_distance(p2, g2, d, sp))
    site = "lung" if "THB" in pid else "abd"
    line = f"  {pid} ({site}): " + "  ".join(f"C{int(c*1000)/10}%={np.nanmean(percp[c]):.4f}" for c in CUTOFFS)
    print(line, flush=True)
    for c in CUTOFFS: agg[c].append(np.nanmean(percp[c]))
print(f">>> MARGIN={MARGIN} mean IDD by cutoff: " + "  ".join(f"C{c*100:.1f}%={np.nanmean(agg[c]):.4f}" for c in CUTOFFS), flush=True)
