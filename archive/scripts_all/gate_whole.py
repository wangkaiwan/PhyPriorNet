"""WHOLE-IMAGE-inference WEPL gate — for models trained WHOLE-image (INSTANCE norm must see the whole
volume; sliding-window on a whole-trained model = the 11.86 trap). Compares, all whole-image inference:
  B      = ref_samefield_aug (2mm whole, deployed)          -> 1.889 known
  W2mm   = ref_2mm_samefield_whole (2mm whole, same aug)    [arg2, optional]
  W13    = ref_1x1x3_samefield_whole (1x1x3 whole, NEW)     [arg1]
Clean whole-regime resolution test: does whole-1x1x3 reach/beat whole-2mm?"""
import os, sys, json, numpy as np, SimpleITK as sitk, torch
sys.path.insert(0, "scripts")
from train_sct_paired import norm_mr, load_arr
from train_sct_refiner import model as refmodel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doserad.physics.density import hu_to_density
import torch.nn.functional as F

CT_LO, CT_HI = -1000.0, 2000.0
def to01(hu): return np.clip((hu - CT_LO)/(CT_HI-CT_LO), 0, 1).astype(np.float32)
def _pad16(n): return (16 - n % 16) % 16
ent = json.load(open("/data/kwang/DoseRad2026_raw/beam_parameters.json"))["hu_to_density"]["entries"]
ANCH = tuple(sorted((float(e["hu"]), float(e["density_g_cm3"])) for e in ent))
val = json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_1x1x3_samefield.json"))["val"]
d2 = json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_2mm_samefield.json"))
by2 = {it["pid"]: it for it in d2["train"] + d2["val"]}

def wepl(sct_hu, ct_hu):
    ds = hu_to_density(sct_hu, ANCH); dc = hu_to_density(ct_hu, ANCH); body = ct_hu > -500
    return float(np.abs(ds.sum(0)-dc.sum(0))[body.any(0)].mean()), float(np.abs(ds-dc)[body].mean()), \
           float(np.abs(ds-dc)[ct_hu>=150].mean())

@torch.no_grad()
def whole_infer(net, mr, co):   # mr, co: (x,y,z) float32 -> sct HU (z,y,x)
    x = torch.from_numpy(np.stack([mr, co], 0)[None]).cuda()
    X, Y, Z = x.shape[-3:]
    xp = F.pad(x, (0, _pad16(Z), 0, _pad16(Y), 0, _pad16(X)))
    with torch.autocast("cuda"):
        p = net(xp)[0, 0, :X, :Y, :Z].float().cpu().numpy()
    return (np.clip(p, 0, 1)*(CT_HI-CT_LO)+CT_LO).transpose(2, 1, 0)

def run_native(ckpt, coarse_dir):   # 1x1x3 whole: no resample (already native)
    net = refmodel().cuda().eval(); net.load_state_dict(torch.load(ckpt, map_location="cpu")["net"])
    R = []
    for it in val:
        mr = norm_mr(load_arr(it["mr"])).astype(np.float32)
        co = to01(load_arr(f"{coarse_dir}/{it['pid']}.nii.gz").astype(np.float32))
        sct = whole_infer(net, mr, co)
        ct = sitk.GetArrayFromImage(sitk.ReadImage(it["ct"])).astype(np.float32)
        R.append(wepl(sct, ct))
    return np.mean(R, 0)

def run_2mm(ckpt, coarse_dir):   # 2mm whole: resample sCT 2mm->native
    net = refmodel().cuda().eval(); net.load_state_dict(torch.load(ckpt, map_location="cpu")["net"])
    R = []
    for it in val:
        it2 = by2[it["pid"]]; mimg = sitk.ReadImage(it2["mr"])
        mr = norm_mr(load_arr(it2["mr"])).astype(np.float32)
        co = to01(load_arr(f"{coarse_dir}/{it['pid']}.nii.gz").astype(np.float32))
        sct2 = whole_infer(net, mr, co)
        im = sitk.GetImageFromArray(sct2); im.CopyInformation(mimg)
        ctimg = sitk.ReadImage(it["ct"])
        sn = sitk.Resample(im, ctimg, sitk.Transform(), sitk.sitkLinear, CT_LO, sitk.sitkFloat32)
        R.append(wepl(sitk.GetArrayFromImage(sn).astype(np.float32), sitk.GetArrayFromImage(ctimg).astype(np.float32)))
    return np.mean(R, 0)

if __name__ == "__main__":
    W13 = sys.argv[1] if len(sys.argv) > 1 else "/data/kwang/sct_refine_runs/ref_1x1x3_samefield_whole/best.pt"
    SF = "/data/kwang/doserad_cache_archive/coarse_ct_whole_soft_samefield"
    print(f"=== WHOLE-IMAGE WEPL gate ({len(val)} val) ===")
    b = run_2mm("/data/kwang/sct_refine_runs/ref_samefield_aug/best.pt", SF)
    print(f"  B    2mm whole deployed   WEPL {b[0]:.3f}  dens {b[1]:.4f}  bone {b[2]:.4f}")
    w2 = "/data/kwang/sct_refine_runs/ref_2mm_samefield_whole/best.pt"
    if os.path.exists(w2):
        c2 = run_2mm(w2, SF); print(f"  W2mm 2mm whole same-aug   WEPL {c2[0]:.3f}  dens {c2[1]:.4f}  bone {c2[2]:.4f}")
    w = run_native(W13, "/data/kwang/coarse_ct_1x1x3_samefield_soft")
    print(f"  W13  1x1x3 whole (NEW)    WEPL {w[0]:.3f}  dens {w[1]:.4f}  bone {w[2]:.4f}")
    print(f"\n>>> whole-1x1x3 {w[0]:.3f} vs 2mm-whole-deployed {b[0]:.3f} — {'BEATS/TIES' if w[0]<=b[0]+0.1 else 'LOSES'} (Δ {w[0]-b[0]:+.3f})")
