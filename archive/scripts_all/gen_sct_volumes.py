"""Generate sCT.mha volumes for the 16 val patients from a trained sCT model, on the cyclegan/ROOT
grid (so eval_sct_route.py --sct-dir can score them through the v13ft dose engine -> plan gamma).
Saves <out_dir>/<pid>/sCT.mha (same .mha format as VBoussot). Outside-body set to air (-1000 HU)."""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from scripts.train_sct_paired import norm_mr, denorm_ct, load_arr, build_model, apply_body_mask_hu

VAL = "/data/kwang/cyclegan_data_2mm/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resunet", "stunet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patch", type=int, nargs=3, required=True)
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--mr_root", default=None, help="if set, read MR from <mr_root>/<pid>/image/mr.mha for all pids")
    a = ap.parse_args()
    dev = "cuda"

    if a.arch == "stunet":
        from scripts.stunet_model import build_stunet
        base = build_stunet("base", deep_supervision=False).to(dev)
        base.load_state_dict(torch.load(a.ckpt, map_location=dev)["net"]); base.eval()
        net = lambda x: torch.sigmoid(base(x))
    else:
        net = build_model().to(dev)
        net.load_state_dict(torch.load(a.ckpt, map_location=dev)["net"]); net.eval()

    if a.mr_root:
        imgs = sorted(glob.glob(os.path.join(a.mr_root, "*", "image", "mr.mha")))
    else:
        imgs = sorted(glob.glob(os.path.join(VAL, "images", "*.nii.gz")))
    for ip in imgs:
        pid = ip.split(os.sep)[-3] if a.mr_root else os.path.basename(ip).replace(".nii.gz", "")
        mr = norm_mr(load_arr(ip))
        with torch.no_grad(), torch.autocast("cuda"):
            y = sliding_window_inference(torch.from_numpy(mr[None, None]).to(dev),
                                         tuple(a.patch), a.sw_batch, net, overlap=0.25, mode="gaussian")
        sct_hu = denorm_ct(y.float().clamp(0, 1).squeeze().cpu().numpy())
        sct_hu, _ = apply_body_mask_hu(sct_hu, mr)            # outside body -> air
        ref = sitk.ReadImage(ip)                              # cyclegan MR geometry (== ROOT grid)
        out = sitk.GetImageFromArray(sct_hu.transpose(2, 1, 0).astype(np.float32))  # (x,y,z)->(z,y,x)
        out.CopyInformation(ref)
        od = os.path.join(a.out_dir, pid); os.makedirs(od, exist_ok=True)
        sitk.WriteImage(out, os.path.join(od, "sCT.mha"))
        print(f"  {pid}: sCT.mha {sct_hu.shape} HU[{sct_hu.min():.0f},{sct_hu.max():.0f}]", flush=True)
    print("DONE ->", a.out_dir, flush=True)


if __name__ == "__main__":
    main()
