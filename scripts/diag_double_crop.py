"""Debug the double-Gaussian eval HANG (26GB, 0% util). Hypothesis: the wide sigma2 halo makes
_tight_from_pb (crop where pb > 1%*max) explode -> huge net crop -> OOM/hang. For one proton patient,
build a few beamlets' PB prior SINGLE vs DOUBLE (via build_ray), run _tight_from_pb, and print the tight
crop shape/voxels for each. If double crops are much bigger, confirmed -> fix = raise the crop threshold
for the halo (far low-dose halo is <10% Dmax, not gamma-evaluated). Also check for NaN/inf in the prior.
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/diag_double_crop.py
"""
import os, sys, json
from pathlib import Path
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DOSERAD_2PASS"] = "0"
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
import numpy as np, torch, SimpleITK as sitk
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.physics.density import hu_to_density
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from accel.proton_build_ray import build_ray
from container.proton.geom_bbox import geom_bbox_proton
from container.proton.predict import _tight_from_pb

DEV = "cuda"; PROT = "/data/kwang/DoseRad2026_raw/proton/training"
pid = os.environ.get("DOSERAD_DIAG_PID") or json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"][0]
machine = ProtonMachineData(device=DEV)
_ent = json.load(open(os.environ["DOSERAD_MACHINE"]))["hu_to_density"]["entries"]
ANCH = tuple(sorted((float(e["hu"]), float(e["density_g_cm3"])) for e in _ent))
class _Img:
    def __init__(s, ct, d): s.array = d; s.spacing = ct.GetSpacing(); s.origin = ct.GetOrigin()
ct = sitk.ReadImage(f"{PROT}/{pid}/image/ct.mha"); hu = sitk.GetArrayFromImage(ct).astype(np.float32)
dens = hu_to_density(hu, ANCH).astype(np.float32); dens_t = torch.as_tensor(dens, device=DEV)
image = _Img(ct, dens)
plan = json.load(open(f"{PROT}/{pid}/{pid}.json"))
b = plan["beams"][0]; r = b["rays"][0]
bls = []
for bl in r["beamlets"][:5]:
    gb = geom_bbox_proton(image.array, image.spacing, image.origin, r["ray_source"], r["ray_target"], machine, bl["energy"])
    if gb is not None: bls.append(dict(energy=bl["energy"], bbox=gb))
print(f"[crop-dbg] pid={pid} beam0 ray0, {len(bls)} beamlets", flush=True)
for mode in ("0", "1"):
    os.environ["DOSERAD_LATERAL_DOUBLE"] = mode
    stacks = build_ray(image, r["ray_source"], r["ray_target"], bls, machine=machine, density=dens_t, device=DEV)
    for (stack, gbb), bl in zip(stacks, bls):
        pb = stack[2] * float(_P_CH_SCALE_PRIOR[2]) / PROTON_DOSE_SCALE
        nan = bool(torch.isnan(pb).any() or torch.isinf(pb).any())
        tb = _tight_from_pb(pb)
        if tb is None: print(f"  {'DOUBLE' if mode=='1' else 'single'} E{bl['energy']:.0f}: tight=None"); continue
        z0,z1,y0,y1,x0,x1 = tb; vox = (z1-z0+1)*(y1-y0+1)*(x1-x0+1)
        print(f"  {'DOUBLE' if mode=='1' else 'single'} E{bl['energy']:.0f}: geom {tuple(pb.shape)} -> tight ({z1-z0+1},{y1-y0+1},{x1-x0+1}) = {vox/1e6:.2f}M vox | pbmax {float(pb.max()):.2e} nan={nan}", flush=True)
print(">>> if DOUBLE tight vox >> single, the halo blows up the crop -> raise _THRESH for double", flush=True)
