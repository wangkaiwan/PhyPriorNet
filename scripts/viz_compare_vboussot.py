"""3-way sCT comparison: ours (paired-supervised UNet) vs VBoussot SynthRAD2025 vs real CT.
Same 2mm grid for all three (verified). In-body HU MAE per method in the title."""
import argparse, os, sys
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_sct_paired import norm_mr, norm_ct, denorm_ct, load_arr, build_model, apply_body_mask_hu
from monai.inferers import sliding_window_inference

VB_DIR = "/data/kwang/synthrad_run/Predictions/Out/Dataset"
VAL = "/data/kwang/cyclegan_data_2mm/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--patient", default="1ABB006")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--arch", choices=["resunet", "stunet"], default="resunet")
    ap.add_argument("--mask_air", action="store_true", help="set sCT outside MR body mask to air")
    a = ap.parse_args()
    p = a.patient
    dev = "cuda"

    mr = norm_mr(load_arr(f"{VAL}/images/{p}.nii.gz"))
    ct_hu = denorm_ct(norm_ct(load_arr(f"{VAL}/labels/{p}.nii.gz")))   # clipped to [-1000,2000]
    vb_hu = load_arr(f"{VB_DIR}/{p}/sCT.mha")

    if a.arch == "stunet":
        from scripts.stunet_model import build_stunet
        base = build_stunet("base", deep_supervision=False).to(dev)
        base.load_state_dict(torch.load(a.ckpt, map_location=dev)["net"])
        net = lambda x: torch.sigmoid(base(x))   # STU-Net: DS off + sigmoid -> [0,1]
        base.eval()
    else:
        net = build_model().to(dev)
        net.load_state_dict(torch.load(a.ckpt, map_location=dev)["net"])
        net.eval()
    with torch.no_grad(), torch.autocast("cuda"):
        y = sliding_window_inference(torch.from_numpy(mr[None, None]).to(dev),
                                     tuple(a.patch), 2, net, overlap=0.25, mode="gaussian")
    ours_hu = denorm_ct(y.float().clamp(0, 1).squeeze().cpu().numpy())
    if a.mask_air:
        ours_hu, _ = apply_body_mask_hu(ours_hu, mr)

    body = ct_hu > -500
    mae_ours = float(np.abs(ours_hu[body] - ct_hu[body]).mean())
    mae_vb = float(np.abs(np.clip(vb_hu, -1000, 2000)[body] - ct_hu[body]).mean())

    z = mr.shape[2] // 2; cy = mr.shape[1] // 2
    cols = [("MRI", mr, 0, 1, "gray"),
            (f"ours sCT", ours_hu, -200, 200, "gray"),
            (f"VBoussot sCT", vb_hu, -200, 200, "gray"),
            ("real CT", ct_hu, -200, 200, "gray"),
            ("ours - CT", ours_hu - ct_hu, -300, 300, "bwr"),
            ("VB - CT", vb_hu - ct_hu, -300, 300, "bwr")]
    fig, ax = plt.subplots(2, 6, figsize=(22, 7.5))
    for r, (rn, sl) in enumerate([("axial", lambda v: v[:, :, z]), ("coronal", lambda v: v[:, cy, :])]):
        for c, (t, vol, vmn, vmx, cm) in enumerate(cols):
            h = ax[r, c].imshow(np.rot90(sl(vol)), vmin=vmn, vmax=vmx, cmap=cm)
            ax[r, c].set_title(f"{rn} {t}", fontsize=9); ax[r, c].axis("off")
            fig.colorbar(h, ax=ax[r, c], fraction=0.046)
    fig.suptitle(f"{p}   in-body MAE:  OURS {mae_ours:.1f} HU   vs   VBoussot {mae_vb:.1f} HU", fontsize=14)
    plt.tight_layout(); plt.savefig(a.out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"saved {a.out} | OURS {mae_ours:.1f} HU  VBoussot {mae_vb:.1f} HU")


if __name__ == "__main__":
    main()
