"""STAGE B GATE: does the 1x1x3-retrained sCT front-end (clf_1x1x3 + ref_1x1x3) beat the 2mm path?
Measures sCT-vs-realCT density error + WEPL on the 16 held-out (fold_0) proton val patients.
  C = new native 1x1x3 cascade (coarse_ct_1x1x3_soft + ref_1x1x3), held-out.
  B = 2mm standalone refiner (coarse_ct_whole_soft_samefield + ref_samefield_aug), same WEPL code.
  A = production 2mm via container synth_density = 1.998 WEPL (measured separately, reference only).
GATE PASSES iff C_WEPL < B_WEPL (and ideally < 1.998). Real CT (native) is ground truth for both."""
import os, sys, json, numpy as np, torch, SimpleITK as sitk
sys.path.insert(0, "scripts")
from train_sct_paired import norm_mr, load_arr
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doserad.physics.density import hu_to_density

from train_sct_refiner import model as refmodel_leaky   # LEAKYRELU (current trainer -> ref_1x1x3)
CT_LO, CT_HI = -1000.0, 2000.0
def to01(hu): return np.clip((hu - CT_LO) / (CT_HI - CT_LO), 0.0, 1.0).astype(np.float32)
def refnet_new():   # matches ref_1x1x3 (current trainer, LEAKYRELU)
    return refmodel_leaky().to("cuda").eval()
def refnet_prelu(): # matches ref_samefield_aug (older default PReLU act)
    return UNet(spatial_dims=3, in_channels=2, out_channels=1, channels=(32,64,128,256,320),
                strides=(2,2,2,2), num_res_units=2, norm="INSTANCE").to("cuda").eval()
BEAM = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
ent = json.load(open(BEAM))["hu_to_density"]["entries"]
ANCH = tuple(sorted((float(e["hu"]), float(e["density_g_cm3"])) for e in ent))

d13 = json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_1x1x3.json"))
val = d13["val"]
d2 = json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_2mm.json"))
by2 = {it["pid"]: it for it in d2["train"] + d2["val"]}

def wepl_row(sct_hu_zyx, ct_hu_zyx):
    ds = hu_to_density(sct_hu_zyx, ANCH); dc = hu_to_density(ct_hu_zyx, ANCH)
    body = ct_hu_zyx > -500; ad = np.abs(ds - dc)
    col = body.any(0)
    return {"dens": float(ad[body].mean()), "bone": float(ad[ct_hu_zyx >= 150].mean()),
            "WEPL": float(np.abs(ds.sum(0) - dc.sum(0))[col].mean())}

@torch.no_grad()
def run_C(ckpt):
    net = refnet_new(); net.load_state_dict(torch.load(ckpt, map_location="cpu")["net"])
    rows = []
    for it in val:
        pid = it["pid"]
        mr = norm_mr(load_arr(it["mr"])).astype(np.float32)                     # (x,y,z)
        coarse = to01(load_arr(f"{os.environ.get('DOSERAD_C_COARSE','/data/kwang/coarse_ct_1x1x3_samefield_soft')}/{pid}.nii.gz").astype(np.float32))
        x = torch.from_numpy(np.stack([mr, coarse], 0)[None]).to("cuda")
        with torch.autocast("cuda"):
            p = sliding_window_inference(x, (128,128,128), 2, net, overlap=0.25, mode="gaussian")[0,0].float().cpu().numpy()
        sct_hu = (np.clip(p,0,1)*(CT_HI-CT_LO)+CT_LO).transpose(2,1,0)          # -> (z,y,x)
        ct_hu = sitk.GetArrayFromImage(sitk.ReadImage(it["ct"])).astype(np.float32)
        rows.append(wepl_row(sct_hu, ct_hu))
    return rows

@torch.no_grad()
def run_B(ckpt, coarse_dir):
    net = refnet_new(); net.load_state_dict(torch.load(ckpt, map_location="cpu")["net"])
    def pad16(n): return (16 - n % 16) % 16
    rows = []
    for it in val:
        pid = it["pid"]; it2 = by2[pid]
        mr_img = sitk.ReadImage(it2["mr"])
        mr = norm_mr(load_arr(it2["mr"])).astype(np.float32)
        coarse = to01(load_arr(f"{coarse_dir}/{pid}.nii.gz").astype(np.float32))
        x = torch.from_numpy(np.stack([mr, coarse], 0)[None]).to("cuda")
        X, Y, Z = x.shape[-3:]
        xp = torch.nn.functional.pad(x, (0, pad16(Z), 0, pad16(Y), 0, pad16(X)))
        with torch.autocast("cuda"):
            p = net(xp)[0,0,:X,:Y,:Z].float().cpu().numpy()
        sct_hu_2mm = (np.clip(p,0,1)*(CT_HI-CT_LO)+CT_LO).transpose(2,1,0)       # (z,y,x) on 2mm grid
        sct_img = sitk.GetImageFromArray(sct_hu_2mm); sct_img.CopyInformation(mr_img)
        ct_img = sitk.ReadImage(it["ct"])                                        # native
        sct_n = sitk.Resample(sct_img, ct_img, sitk.Transform(), sitk.sitkLinear, CT_LO, sitk.sitkFloat32)
        sct_hu = sitk.GetArrayFromImage(sct_n).astype(np.float32)
        ct_hu = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        rows.append(wepl_row(sct_hu, ct_hu))
    return rows

def summ(tag, rows):
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"  {tag:28s} dens {m['dens']:.4f}  bone {m['bone']:.4f}  WEPL {m['WEPL']:.4f}")
    return m

if __name__ == "__main__":
    C_ckpt = sys.argv[1] if len(sys.argv) > 1 else "/data/kwang/sct_refine_runs/ref_1x1x3_samefield/best.pt"
    print(f"=== FAIR 1x1x3 REDO GATE ({len(val)} val) — sCT-vs-realCT WEPL ===")
    B  = summ("B  2mm deployed (ref_samefield_aug)", run_B("/data/kwang/sct_refine_runs/ref_samefield_aug/best.pt",
                                              "/data/kwang/doserad_cache_archive/coarse_ct_whole_soft_samefield"))
    B2 = summ("B2 2mm control (same aug, 182)", run_B("/data/kwang/sct_refine_runs/ref_2mm_samefield_curaug/best.pt",
                                              "/data/kwang/doserad_cache_archive/coarse_ct_whole_soft_samefield")) \
         if os.path.exists("/data/kwang/sct_refine_runs/ref_2mm_samefield_curaug/best.pt") else None
    C  = summ("C  1x1x3 (NEW, ref_1x1x3_samefield)", run_C(C_ckpt))
    print(f"\n>>> WEPL:  C(1x1x3) {C['WEPL']:.3f}   vs   B(2mm deployed) {B['WEPL']:.3f}"
          + (f"   vs   B2(2mm same-aug) {B2['WEPL']:.3f}" if B2 else ""))
    ref = min([x['WEPL'] for x in [B, B2] if x])
    print(">>> VERDICT (resolution net effect):",
          f"1x1x3 {'BEATS' if C['WEPL']<ref else 'LOSES to'} best 2mm ({C['WEPL']:.3f} vs {ref:.3f}) — Δ {C['WEPL']-ref:+.3f}")
