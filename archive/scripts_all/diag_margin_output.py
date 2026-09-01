"""Verify the margin change is CLEAN: predict the SAME photon-CT patient at margin-24 and margin-8,
and check that within the margin-8 crop the dose is IDENTICAL (fully-conv => no core change), so the
only difference is the cropped scatter tail. If the core differs, the margin plumbing has a bug.
Runs the deployed model (4018f597) twice via container.photon.app with different DOSERAD_PHOTON_MARGIN.
Usage: CUDA_VISIBLE_DEVICES=1 python scripts/diag_margin_output.py [pid]
"""
import os, sys, json, importlib
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DOSERAD_WEIGHTS"] = "/home/kaiwang/doserad2026_workdir/runs/docker_extracted/photon_ct_docker.pt"
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
import numpy as np, SimpleITK as sitk
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for cp in b["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return {"image_file_idx": 0, "beams": beams}

def predict_at(margin):
    os.environ["DOSERAD_PHOTON_MARGIN"] = str(margin)
    import container.photon.predict as P; importlib.reload(P)          # re-read _MARGIN
    import container.photon.app as app; importlib.reload(app)
    app.load_models()
    src = sitk.ReadImage(f"{PHOTON_ROOT}/{PID}/image/ct.mha")
    plan = json.load(open(f"{PHOTON_ROOT}/{PID}/{PID}.json"))
    return app._predict_fn(src, build_entry(plan)), sitk.GetArrayFromImage(src).shape

def embed(crop, bb, full):
    z0,z1,y0,y1,x0,x1 = bb; a = np.zeros(full, np.float64); a[z0:z1+1,y0:y1+1,x0:x1+1] = crop; return a

p24, full = predict_at(24)
p8, _ = predict_at(8)
print(f"[margin-diff] pid={PID}  comparing margin-24 vs margin-8 output per CP", flush=True)
keys = sorted(set(p24) & set(p8))[:6]
for k in keys:
    d24 = embed(p24[k][0].astype(np.float64), p24[k][1], full)
    d8  = embed(p8[k][0].astype(np.float64),  p8[k][1],  full)
    core = d8 > 0                                   # the margin-8 footprint
    core_maxdiff = float(np.abs(d24[core] - d8[core]).max()) if core.any() else 0.0
    core_rel = core_maxdiff / max(float(d24.max()), 1e-12)
    tail_energy = float(d24[~core].sum())           # dose in the cropped (margin-8..24) region
    tot24 = float(d24.sum())
    print(f"  CP{k}: core max|Δ| {core_maxdiff:.3e} (rel {core_rel:.2e})  |  cropped-tail = {tail_energy/max(tot24,1e-12)*100:.2f}% of CP dose", flush=True)
print(">>> if core max|Δ|~0, margin only crops the tail (CLEAN, no bug); the tail% is what margin-8 drops.", flush=True)
