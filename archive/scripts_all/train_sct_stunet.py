"""STU-Net paired MR->sCT training (the 'high-end' version). Reuses the dataset / aug / normalization
/ dose-weight from train_sct_paired.py, swaps the model to STU-Net (base, ~58M, deep supervision),
patch 192x192x64, MAE loss with dual-end (bone+lung) dose-sensitivity weighting applied at every
deep-supervision scale. Output via sigmoid -> [0,1] (matches STU-Net 0-1 range + our [0,1] norm).

Eval metrics: in-body HU MAE, SSIM, Pearson corr. Best ckpt + early stop are driven by (SSIM + corr)
(higher = better). LR halves on plateau of that same score.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.metrics import SSIMMetric

from scripts.train_sct_paired import (
    norm_mr, norm_ct, denorm_ct, load_arr, PairedSCT, dose_weight,
    apply_body_mask_hu,
)
from scripts.stunet_model import build_stunet


def ds_weighted_mae(outputs, target, bone_w, lung_w):
    """Deep-supervision MAE: sigmoid each scale output, downsample target to match, dual-end
    dose-weighted L1, combine with 1/2^i weights (normalized; coarser scales matter less)."""
    weights = [1.0 / (2 ** i) for i in range(len(outputs))]
    s = sum(weights)
    loss = 0.0
    for o, w in zip(outputs, weights):
        pred = torch.sigmoid(o)
        tgt = target if pred.shape[2:] == target.shape[2:] else \
            F.interpolate(target, size=pred.shape[2:], mode="trilinear", align_corners=False)
        wmap = dose_weight(tgt, bone_w, lung_w)
        loss = loss + (w / s) * ((wmap * (pred - tgt).abs()).sum() / wmap.sum())
    return loss


class SigmoidInfer(torch.nn.Module):
    """Wrap STU-Net for inference: deep supervision off -> single tensor, then sigmoid -> [0,1]."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


@torch.no_grad()
def validate(net, val_dir, patch, device, amp=True):
    ssim_m = SSIMMetric(spatial_dims=3, data_range=1.0)
    infer = SigmoidInfer(net)
    prev_ds = net.decoder.deep_supervision
    net.decoder.deep_supervision = False
    net.eval()
    maes, ssims, corrs = [], [], []
    for ip in sorted(glob.glob(os.path.join(val_dir, "images", "*.nii.gz"))):
        lp = ip.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        mr = norm_mr(load_arr(ip)); ct = norm_ct(load_arr(lp))
        x = torch.from_numpy(mr[None, None]).to(device)
        with torch.autocast("cuda", enabled=amp):
            y = sliding_window_inference(x, patch, 2, infer, overlap=0.25, mode="gaussian")
        y = y.float().clamp(0, 1)
        ct_t = torch.from_numpy(ct[None, None]).to(device)
        ssims.append(float(ssim_m(y, ct_t).mean().item()))
        sct_hu = denorm_ct(y.squeeze().cpu().numpy()); ct_hu = denorm_ct(ct)
        body = ct_hu > -500
        if body.sum() > 100:
            a, b = sct_hu[body], ct_hu[body]
            maes.append(float(np.abs(a - b).mean()))
            if a.std() > 0 and b.std() > 0:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    net.decoder.deep_supervision = prev_ds
    net.train()
    return (float(np.mean(maes)), float(np.mean(ssims)), float(np.mean(corrs)))


@torch.no_grad()
def save_fig(net, val_dir, patch, device, out_png, patient=None, amp=True):
    infer = SigmoidInfer(net)
    prev_ds = net.decoder.deep_supervision
    net.decoder.deep_supervision = False
    net.eval()
    imgs = sorted(glob.glob(os.path.join(val_dir, "images", "*.nii.gz")))
    ip = imgs[0] if patient is None else os.path.join(val_dir, "images", patient + ".nii.gz")
    lp = ip.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    name = os.path.basename(ip).replace(".nii.gz", "")
    mr = norm_mr(load_arr(ip)); ct_hu = denorm_ct(norm_ct(load_arr(lp)))
    x = torch.from_numpy(mr[None, None]).to(device)
    with torch.autocast("cuda", enabled=amp):
        y = sliding_window_inference(x, patch, 2, infer, overlap=0.25, mode="gaussian")
    sct_hu = denorm_ct(y.float().clamp(0, 1).squeeze().cpu().numpy())
    sct_hu, _ = apply_body_mask_hu(sct_hu, mr)
    net.decoder.deep_supervision = prev_ds
    net.train()
    diff = sct_hu - ct_hu
    z = mr.shape[2] // 2; cy = mr.shape[1] // 2
    body = ct_hu > -500
    mae = float(np.abs(diff[body]).mean())
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for r, (rn, sl) in enumerate([("axial", lambda v: v[:, :, z]), ("coronal", lambda v: v[:, cy, :])]):
        panels = [("MRI", mr, 0, 1, "gray"), ("sCT (HU)", sct_hu, -200, 200, "gray"),
                  ("CT (HU)", ct_hu, -200, 200, "gray"), ("sCT-CT", diff, -300, 300, "bwr")]
        for c, (t, vol, vmn, vmx, cm) in enumerate(panels):
            h = ax[r, c].imshow(np.rot90(sl(vol)), vmin=vmn, vmax=vmx, cmap=cm)
            ax[r, c].set_title(f"{rn} {t}", fontsize=9); ax[r, c].axis("off")
            fig.colorbar(h, ax=ax[r, c], fraction=0.046)
    fig.suptitle(f"{name}  in-body MAE {mae:.1f} HU", fontsize=13)
    plt.tight_layout(); plt.savefig(out_png, dpi=100, bbox_inches="tight"); plt.close(fig)
    return mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/kwang/cyclegan_data_2mm/train")
    ap.add_argument("--val_dir", default="/data/kwang/cyclegan_data_2mm/test")
    ap.add_argument("--out", default="/data/kwang/sct_paired_runs/stunet_v1")
    ap.add_argument("--name", default="sct_stunet_v1")
    ap.add_argument("--variant", default="base", choices=["small", "base"])
    ap.add_argument("--patch", type=int, nargs=3, default=[192, 192, 64])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--iters_per_epoch", type=int, default=120)
    ap.add_argument("--bone_weight", type=float, default=5.0)
    ap.add_argument("--lung_weight", type=float, default=3.0)
    ap.add_argument("--lr_patience", type=int, default=15)
    ap.add_argument("--early_stop", type=int, default=40, help="stop after N epochs w/o (ssim+corr) improvement")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--viz_every", type=int, default=10)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    patch = tuple(a.patch); device = "cuda"

    import wandb
    wandb.init(project="doserad2026", name=a.name, config=vars(a), dir=a.out)

    ds = PairedSCT(a.data_dir, patch, train=True, cache=False, aggressive=True)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    sampler=torch.utils.data.RandomSampler(ds, replacement=True,
                                                           num_samples=a.iters_per_epoch * a.batch))

    net = build_stunet(a.variant, deep_supervision=True).to(device)
    print("STU-Net params (M):", round(sum(p.numel() for p in net.parameters()) / 1e6, 2), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=a.lr_patience, threshold=1e-3, min_lr=a.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda")

    best_score = -1e9; best_mae = 1e9; since_improve = 0
    for epoch in range(1, a.epochs + 1):
        net.train(); t0 = time.time(); run = 0.0; n = 0
        for mr, ct in dl:
            mr = mr.to(device, non_blocking=True); ct = ct.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                outs = net(mr)
                loss = ds_weighted_mae(outs, ct, a.bone_weight, a.lung_weight)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss.detach()) * mr.size(0); n += mr.size(0)
        tr_loss = run / max(n, 1)

        mae, ssim, corr = validate(net, a.val_dir, patch, device)
        score = ssim + corr
        sched.step(score)
        log = {"train/loss": tr_loss, "val/mae_hu": mae, "val/ssim": ssim, "val/corr": corr,
               "val/score": score, "lr": opt.param_groups[0]["lr"], "epoch": epoch}
        print(f"epoch {epoch}: loss {tr_loss:.4f}  MAE {mae:.1f}  SSIM {ssim:.4f}  corr {corr:.4f}  "
              f"score {score:.4f}  lr {opt.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)", flush=True)

        torch.save({"net": net.state_dict(), "epoch": epoch}, os.path.join(a.out, "last.pt"))
        if score > best_score + 1e-4:
            best_score = score; best_mae = mae; since_improve = 0
            torch.save({"net": net.state_dict(), "epoch": epoch, "mae": mae, "ssim": ssim, "corr": corr},
                       os.path.join(a.out, "best.pt"))
            print(f"  ** new BEST score {score:.4f} (MAE {mae:.1f} SSIM {ssim:.4f} corr {corr:.4f}) -> best.pt **", flush=True)
        else:
            since_improve += 1

        if epoch % a.viz_every == 0:
            fp = os.path.join(a.out, "figs", f"epoch_{epoch:04d}.png")
            save_fig(net, a.val_dir, patch, device, fp)
        wandb.log(log, step=epoch)

        if since_improve >= a.early_stop:
            print(f"EARLY STOP at epoch {epoch}: no (ssim+corr) improvement for {a.early_stop} epochs.", flush=True)
            break
    print(f"DONE. best score {best_score:.4f} (MAE {best_mae:.1f} HU)", flush=True)


if __name__ == "__main__":
    main()
