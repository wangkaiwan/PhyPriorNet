"""Paired supervised MR->sCT training (3D), the replacement for the unpaired CycleGAN route.

Data: SynthRAD2025-derived registered MR/CT pairs at 2mm (images=MR, labels=CT, voxel-aligned).
  train: /data/kwang/cyclegan_data_2mm/train   (282 leak-free AB+TH pairs)
  val  : /data/kwang/cyclegan_data_2mm/test     (16 DoseRAD val patients)

Model : MONAI residual 3D UNet (regression). Loss: L1 + lambda_ssim * (1 - SSIM).
Norm  : MR per-patient [p1,p99] -> [0,1];  CT HU clip [-1000,2000] -> [0,1] (invertible).
Train : random foreground patches + flip/noise aug, AdamW + cosine, AMP.
Val   : sliding-window full-volume inference on the 16 val patients; in-body HU MAE (primary,
        lower=better) + SSIM + PSNR. best.pt saved on lowest MAE.
Viz   : every --viz_every epochs, an MR|sCT|CT|diff figure for a fixed val patient is written to
        <out>/figs/epoch_XXXX.png (so progress is visually inspectable).
"""
from __future__ import annotations
import argparse, glob, os, time, random
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.losses import SSIMLoss
from monai.metrics import SSIMMetric
from monai.transforms import (
    Compose, RandFlipd, RandAffined, RandGaussianNoised, RandBiasFieldd,
    RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd, RandGaussianSmoothd,
)


def build_aug():
    """Aggressive paired augmentation (channel-first dict {image: MR, label: CT}, both in [0,1],
    air/bg = 0 so geometric border-fill 0 is correct). Geometric ops apply to BOTH (paired);
    intensity ops apply to MR only (image). Affine pushed harder than VBoussot (they only flip)."""
    geo = ["image", "label"]
    return Compose([
        RandFlipd(keys=geo, spatial_axis=0, prob=0.5),
        RandFlipd(keys=geo, spatial_axis=1, prob=0.5),
        RandFlipd(keys=geo, spatial_axis=2, prob=0.5),
        RandAffined(keys=geo, prob=0.75,
                    rotate_range=(0.26, 0.26, 0.26),     # ~±15 deg each axis
                    scale_range=(0.2, 0.2, 0.2),          # ±20% zoom
                    translate_range=(8, 8, 8),
                    mode=("bilinear", "bilinear"), padding_mode="zeros"),
        # MR-only intensity (sCT target CT stays clean)
        RandBiasFieldd(keys=["image"], prob=0.4, coeff_range=(0.0, 0.12)),   # MR bias field
        RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.7, 1.5)),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.4, 1.0), sigma_y=(0.4, 1.0), sigma_z=(0.4, 1.0)),
        RandGaussianNoised(keys=["image"], prob=0.4, std=0.03),
        RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.12),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.06),
    ])

# ----------------------------- normalization -----------------------------
CT_MIN, CT_MAX = -1000.0, 2000.0   # HU window -> [0,1]; covers air..bone, clips rare metal


def norm_mr(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def norm_ct(a):
    a = a.astype(np.float32)
    return np.clip((a - CT_MIN) / (CT_MAX - CT_MIN), 0.0, 1.0)


def denorm_ct(x):
    return np.asarray(x, np.float32) * (CT_MAX - CT_MIN) + CT_MIN


def load_arr(p):
    return np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32), (2, 1, 0))


def body_mask_from_mr(mr_norm, thr=0.04):
    """Body envelope from a normalized [0,1] MR via simple morphology. CONSERVATIVE by design: the
    mask must be a SUPERSET of the true body (never clip skin/fat), so we use a low threshold, keep
    the largest component, fill holes (lungs/cavities = inside body), then DILATE a margin. Only
    clearly-external air is removed; body MAE is unchanged, spurious air-region values are zeroed."""
    from scipy import ndimage
    m = mr_norm > thr
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab, np.float32), lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_dilation(m, iterations=3)   # safety margin: never clip real body
    return m


def apply_body_mask_hu(sct_hu, mr_norm, air_hu=-1000.0):
    """Set sCT to air outside the MR-derived body mask."""
    m = body_mask_from_mr(mr_norm)
    out = sct_hu.copy()
    out[~m] = air_hu
    return out, m


# ----------------------------- dataset -----------------------------
class PairedSCT(Dataset):
    def __init__(self, data_dir, patch, train=True, cache=True, aggressive=False):
        self.imgs = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
        self.lbls = [p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep) for p in self.imgs]
        self.patch = patch
        self.train = train
        self.cache = {} if cache else None
        self.aug = build_aug() if (train and aggressive) else None

    def __len__(self):
        return len(self.imgs)

    def _get(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        mr = norm_mr(load_arr(self.imgs[i]))
        ct = norm_ct(load_arr(self.lbls[i]))
        if self.cache is not None:
            self.cache[i] = (mr, ct)
        return mr, ct

    def _pad(self, a, fill):
        ph = [max(self.patch[d] - a.shape[d], 0) for d in range(3)]
        if any(ph):
            a = np.pad(a, [(0, ph[0]), (0, ph[1]), (0, ph[2])], constant_values=fill)
        return a

    def __getitem__(self, i):
        mr, ct = self._get(i)
        mr = self._pad(mr, 0.0); ct = self._pad(ct, 0.0)
        ps = self.patch
        # foreground-biased crop: center on a body voxel (ct above air) when possible
        body = ct > norm_ct(np.array([-500.0]))[0]
        st = [0, 0, 0]
        if self.train and body.sum() > 0:
            idx = np.argwhere(body)
            c = idx[random.randrange(len(idx))]
            for d in range(3):
                st[d] = int(np.clip(c[d] - ps[d] // 2, 0, mr.shape[d] - ps[d]))
        else:
            for d in range(3):
                st[d] = (mr.shape[d] - ps[d]) // 2
        sl = tuple(slice(st[d], st[d] + ps[d]) for d in range(3))
        mp, cp = mr[sl].copy(), ct[sl].copy()
        if self.train and self.aug is not None:
            d = self.aug({"image": torch.from_numpy(mp[None]), "label": torch.from_numpy(cp[None])})
            mp_t = torch.as_tensor(d["image"]).float().clamp(0, 1)
            cp_t = torch.as_tensor(d["label"]).float().clamp(0, 1)
            return mp_t, cp_t
        if self.train:  # light aug fallback
            for ax in range(3):
                if random.random() < 0.5:
                    mp = np.flip(mp, ax).copy(); cp = np.flip(cp, ax).copy()
            if random.random() < 0.5:
                mp = np.clip(mp + np.random.normal(0, random.uniform(0.005, 0.02), mp.shape).astype(np.float32), 0, 1)
        return torch.from_numpy(mp[None]), torch.from_numpy(cp[None])


# ----------------------------- loss -----------------------------
# DUAL-END dose-sensitivity weighting. Dose depends on density rho ~ (HU+1000)/1000; a fixed HU error
# perturbs dose far more where density is far from water (lung: low rho; bone: high rho) than near
# water (soft tissue). So we up-weight the L1 at BOTH ends and keep soft tissue (~water) at 1:
#   bone:  weight ramps 1 -> bone_weight over HU [150, 1000]   (cortical bone)
#   lung:  weight ramps 1 -> lung_weight over HU [-300, -800]  (lung parenchyma / low density)
# HU window [-1000,2000]->[0,1] (t), so the breakpoints in normalized t:
_BONE_LO = (150.0 + 1000.0) / 3000.0    # 0.383
_BONE_HI = (1000.0 + 1000.0) / 3000.0   # 0.667
_LUNG_HI = (-300.0 + 1000.0) / 3000.0   # 0.233 (start weighting below this HU)
_LUNG_LO = (-800.0 + 1000.0) / 3000.0   # 0.067 (full lung weight at/below this HU)


def dose_weight(target, bone_weight, lung_weight):
    bone_ramp = ((target - _BONE_LO) / (_BONE_HI - _BONE_LO)).clamp(0.0, 1.0)
    lung_ramp = ((_LUNG_HI - target) / (_LUNG_HI - _LUNG_LO)).clamp(0.0, 1.0)
    return 1.0 + (bone_weight - 1.0) * bone_ramp + (lung_weight - 1.0) * lung_ramp


def weighted_l1(pred, target, bone_weight, lung_weight=1.0):
    if bone_weight <= 1.0 and lung_weight <= 1.0:
        return (pred - target).abs().mean()
    w = dose_weight(target, bone_weight, lung_weight)
    return (w * (pred - target).abs()).sum() / w.sum()


# ----------------------------- model -----------------------------
def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=1,
                channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
                num_res_units=2, norm="INSTANCE", act="LEAKYRELU")


# ----------------------------- validation -----------------------------
@torch.no_grad()
def validate(net, val_dir, patch, device, amp=True):
    ssim_m = SSIMMetric(spatial_dims=3, data_range=1.0)
    imgs = sorted(glob.glob(os.path.join(val_dir, "images", "*.nii.gz")))
    maes, ssims, psnrs = [], [], []
    net.eval()
    for ip in imgs:
        lp = ip.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        mr = norm_mr(load_arr(ip)); ct = norm_ct(load_arr(lp))
        x = torch.from_numpy(mr[None, None]).to(device)
        with torch.autocast("cuda", enabled=amp):
            y = sliding_window_inference(x, patch, 4, net, overlap=0.25, mode="gaussian")
        y = y.float().clamp(0, 1)
        ct_t = torch.from_numpy(ct[None, None]).to(device)
        ssims.append(float(ssim_m(y, ct_t).mean().item()))
        sct_hu = denorm_ct(y.squeeze().cpu().numpy())
        ct_hu = denorm_ct(ct)
        body = ct_hu > -500
        if body.sum() > 100:
            err = np.abs(sct_hu[body] - ct_hu[body])
            maes.append(float(err.mean()))
            mse = float(((sct_hu[body] - ct_hu[body]) ** 2).mean())
            psnrs.append(10 * np.log10((3000.0 ** 2) / max(mse, 1e-6)))
    net.train()
    return (float(np.mean(maes)), float(np.mean(ssims)), float(np.mean(psnrs)))


@torch.no_grad()
def save_fig(net, val_dir, patch, device, out_png, patient=None, amp=True):
    imgs = sorted(glob.glob(os.path.join(val_dir, "images", "*.nii.gz")))
    ip = imgs[0] if patient is None else os.path.join(val_dir, "images", patient + ".nii.gz")
    lp = ip.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    name = os.path.basename(ip).replace(".nii.gz", "")
    mr = norm_mr(load_arr(ip)); ct = norm_ct(load_arr(lp))
    net.eval()
    x = torch.from_numpy(mr[None, None]).to(device)
    with torch.autocast("cuda", enabled=amp):
        y = sliding_window_inference(x, patch, 4, net, overlap=0.25, mode="gaussian")
    net.train()
    sct_hu = denorm_ct(y.float().clamp(0, 1).squeeze().cpu().numpy())
    ct_hu = denorm_ct(ct)
    diff = sct_hu - ct_hu
    z = mr.shape[2] // 2; cy = mr.shape[1] // 2
    body = ct_hu > -500
    mae = float(np.abs(diff[body]).mean())
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    rows = [("axial", lambda v: v[:, :, z]), ("coronal", lambda v: v[:, cy, :])]
    for r, (rn, sl) in enumerate(rows):
        panels = [("MRI", norm_mr(load_arr(ip)), 0, 1, "gray"),
                  ("sCT (HU)", sct_hu, -200, 200, "gray"),
                  ("CT (HU)", ct_hu, -200, 200, "gray"),
                  ("sCT-CT (HU)", diff, -300, 300, "bwr")]
        for c, (t, vol, vmn, vmx, cm) in enumerate(panels):
            h = ax[r, c].imshow(np.rot90(sl(vol)), vmin=vmn, vmax=vmx, cmap=cm)
            ax[r, c].set_title(f"{rn} {t}", fontsize=9); ax[r, c].axis("off")
            fig.colorbar(h, ax=ax[r, c], fraction=0.046)
    fig.suptitle(f"{name}  in-body MAE {mae:.1f} HU", fontsize=13)
    plt.tight_layout(); plt.savefig(out_png, dpi=100, bbox_inches="tight"); plt.close(fig)
    return mae


# ----------------------------- train -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/kwang/cyclegan_data_2mm/train")
    ap.add_argument("--val_dir", default="/data/kwang/cyclegan_data_2mm/test")
    ap.add_argument("--out", default="/data/kwang/sct_paired_runs/v1")
    ap.add_argument("--name", default="sct_paired_v1")
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--iters_per_epoch", type=int, default=140)
    ap.add_argument("--lambda_ssim", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--viz_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--aggressive", action="store_true", help="aggressive MONAI augmentation")
    ap.add_argument("--init_from", default="", help="warm-start: load net weights from this ckpt")
    ap.add_argument("--bone_weight", type=float, default=1.0, help="L1 weight ramp 1->W over HU 150..1000 (bone)")
    ap.add_argument("--lung_weight", type=float, default=1.0, help="L1 weight ramp 1->W over HU -300..-800 (lung)")
    ap.add_argument("--lr_sched", choices=["cosine", "plateau"], default="cosine")
    ap.add_argument("--lr_patience", type=int, default=15, help="plateau: epochs w/o val-MAE improvement before halving")
    ap.add_argument("--lr_factor", type=float, default=0.5, help="plateau: LR multiplier on plateau")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    patch = tuple(a.patch)
    device = "cuda"

    import wandb
    wandb.init(project="doserad2026", name=a.name, config=vars(a), dir=a.out)

    ds = PairedSCT(a.data_dir, patch, train=True, cache=False, aggressive=a.aggressive)  # load-on-the-fly
    # each epoch = iters_per_epoch random patches (sampling volumes with replacement)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    sampler=torch.utils.data.RandomSampler(ds, replacement=True,
                                                           num_samples=a.iters_per_epoch * a.batch))

    net = build_model().to(device)
    if a.init_from:
        sd = torch.load(a.init_from, map_location=device)
        net.load_state_dict(sd["net"] if "net" in sd else sd)
        print(f"warm-started from {a.init_from} (epoch {sd.get('epoch','?')}, mae {sd.get('mae','?')})", flush=True)
    print("params (M):", sum(p.numel() for p in net.parameters()) / 1e6, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    if a.lr_sched == "plateau":
        # user preference: halve LR after `lr_patience` epochs with no val-MAE improvement (mode=min)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=a.lr_factor, patience=a.lr_patience, threshold=1e-3, min_lr=a.lr * 0.01)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda")
    ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0).to(device)

    best_mae = 1e9
    for epoch in range(1, a.epochs + 1):
        net.train(); t0 = time.time(); run = 0.0; n = 0
        for mr, ct in dl:
            mr = mr.to(device, non_blocking=True); ct = ct.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                pred = net(mr)
                pred01 = pred.clamp(0, 1)
                loss = weighted_l1(pred, ct, a.bone_weight, a.lung_weight) + a.lambda_ssim * ssim_loss(pred01, ct)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += float(loss.detach()) * mr.size(0); n += mr.size(0)
        if a.lr_sched == "cosine":
            sched.step()
        tr_loss = run / max(n, 1)
        log = {"train/loss": tr_loss, "lr": opt.param_groups[0]["lr"], "epoch": epoch}

        if epoch % a.val_every == 0:
            mae, ssim, psnr = validate(net, a.val_dir, patch, device)
            log.update({"val/mae_hu": mae, "val/ssim": ssim, "val/psnr": psnr})
            print(f"epoch {epoch}: loss {tr_loss:.4f}  val MAE {mae:.1f}HU  SSIM {ssim:.4f}  PSNR {psnr:.2f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
            if mae < best_mae:
                best_mae = mae
                torch.save({"net": net.state_dict(), "epoch": epoch, "mae": mae}, os.path.join(a.out, "best.pt"))
                print(f"  ** new BEST val MAE {mae:.1f} HU -> saved best.pt **", flush=True)
            if a.lr_sched == "plateau":
                prev = opt.param_groups[0]["lr"]
                sched.step(mae)
                if opt.param_groups[0]["lr"] < prev:
                    print(f"  [lr] plateau: {prev:.2e} -> {opt.param_groups[0]['lr']:.2e}", flush=True)
        torch.save({"net": net.state_dict(), "epoch": epoch}, os.path.join(a.out, "last.pt"))

        if epoch % a.viz_every == 0:
            fp = os.path.join(a.out, "figs", f"epoch_{epoch:04d}.png")
            vmae = save_fig(net, a.val_dir, patch, device, fp)
            log["viz_mae_hu"] = vmae
            print(f"  [fig] {fp}  MAE {vmae:.1f} HU", flush=True)

        wandb.log(log, step=epoch)
    print(f"DONE. best val MAE {best_mae:.1f} HU", flush=True)


if __name__ == "__main__":
    main()
