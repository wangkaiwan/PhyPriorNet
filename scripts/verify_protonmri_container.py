"""Verify the proton-MRI CONTAINER native path reproduces eval_proton_e2e_held16's density EXACTLY.

Container `synth_density(native_grid=True)` runs clf LIVE (clf_1x1x3_samefield) -> coarse -> E2E synth
-> density, on the native 1x1x3 grid. eval_proton_e2e_held16.e2e_density uses PRECOMPUTED coarse
(coarse_ct_1x1x3_samefield_soft). If the two densities match (~0), the container reproduces the exact
input that scored held16 gamma 93.04 -> and since both share container.proton.predict.predict_beams,
the gamma is identical. Prints max|Δ| density per pid; a couple of pids is a deterministic plumbing
check (not a metric estimate).

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/verify_protonmri_container.py <e2e_ckpt.pt> [npid]
"""
import os, sys, json
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, yaml, SimpleITK as sitk

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_protonmri/ckpt_30k.pt"
NPID = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = "configs/experiments/all75/e2e_1x1x3_protonmri.yaml"
CLF_SAMEFIELD = "/data/kwang/sct_classify_runs/clf_1x1x3_samefield/best.pt"
PROT = "/data/kwang/DoseRad2026_raw/proton/training"
cfg = yaml.safe_load(open(CFG))

from train_dose_e2e import E2E
from container.mri_synth import synth_density, load_classifier
import scripts.eval_proton_e2e_held16 as EV   # reuse its e2e_density (precomputed-coarse path)

net = E2E(cfg).to(DEV).eval()
sd = torch.load(CKPT, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
EV.net = net                                    # make eval's e2e_density use THIS ckpt
clf = load_classifier(CLF_SAMEFIELD, DEV)
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"][:NPID]
print(f"[verify] ckpt={CKPT} step={sd.get('step','?')} clf={CLF_SAMEFIELD}", flush=True)

for pid in PIDS:
    mr_sitk = sitk.ReadImage(f"{PROT}/{pid}/image/mr.mha")
    dens_container, _ = synth_density(mr_sitk, clf, net, DEV, density_direct=True, native_grid=True)  # (z,y,x)
    dens_eval = EV.e2e_density(pid)                                                                   # (z,y,x) precomputed coarse
    d = np.abs(dens_container - dens_eval)
    print(f"  {pid}: shape {dens_container.shape} vs {dens_eval.shape} | "
          f"max|Δρ| {d.max():.4f}  mean|Δρ| {d.mean():.5f}  "
          f"(container ρ[{dens_container.min():.2f},{dens_container.max():.2f}])", flush=True)
print(">>> if max|Δρ| ~0 the container native path == the held16-eval input (gamma 93.04 carries over)")
