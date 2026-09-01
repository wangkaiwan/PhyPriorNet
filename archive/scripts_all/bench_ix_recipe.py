"""Timing bench for the "ix recipe" replication (2026-08-25): measure deploy-path ms/CP for
(weights, margin) combos on real patients, to project board runtime.

Board model: T = t_fix + N_cp * t_cp. Our m16 board anchor: 80.3s total, ~540 CPs, t_cp ~0.116s
(5090-warm ~70ms => platform ~116ms => platform/local ratio ~1.66). Projection for a candidate:
T_board ~= t_fix_board (17.7s) + 540 * t_cp_local * 1.66.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/bench_ix_recipe.py <weights.pt> <margin> [n_patients]
Prints warm ms/CP (patient 2+ only; patient 1 includes compile warmup).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
W = sys.argv[1]; MARGIN = sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DOSERAD_PHOTON_MARGIN"] = MARGIN
os.environ["DOSERAD_WEIGHTS"] = W
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
os.environ["DOSERAD_MODALITY"] = "ct"
import SimpleITK as sitk
from container.photon import app

FROZEN = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))["held16"]
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        bb = dict(b); bb["beam_idx"] = bi
        for cp in bb["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(bb)
    return {"image_file_idx": 0, "beams": beams}

app.load_models()
print(f"[bench] weights={W}\n[bench] margin={MARGIN}", flush=True)
warm = []
for i, pid in enumerate(FROZEN[:NP]):
    src = sitk.ReadImage(f"{ROOT}/{pid}/image/ct.mha")
    plan = json.load(open(f"{ROOT}/{pid}/{pid}.json"))
    entry = build_entry(plan)
    ncp = sum(len(b["control_points"]) for b in plan["beams"])
    t0 = time.time()
    preds = app._predict_fn(src, entry)
    import torch; torch.cuda.synchronize()
    dt = time.time() - t0
    tag = "(cold)" if i == 0 else ""
    if i > 0: warm.append(dt * 1000.0 / ncp)
    print(f"  {pid}: {ncp} CPs in {dt:.1f}s = {dt*1000/ncp:.1f} ms/CP {tag}", flush=True)
import numpy as np
if warm:
    ms = float(np.mean(warm))
    proj = 17.7 + 540 * ms / 1000.0 * 1.66
    print(f">>> warm {ms:.1f} ms/CP | board projection ~{proj:.0f}s (t_fix 17.7 + 540*{ms:.1f}ms*1.66)", flush=True)
