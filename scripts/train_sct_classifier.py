"""Lever-4 stage 1: MR -> tissue-class segmentation (air / lung / soft / bone).

Rationale: bone sCT error is a systematic UNDER-estimation bias (ME ~ -345 HU) because cortical bone is
an MR signal void (ambiguous with air) and L1 regression hedges to the soft-tissue median. Classifying
"is bone" is easier than regressing 1200 HU and exploits anatomical context -> a coarse but correctly-
*located* tissue map. From the classes we build a bulk-density / coarse-CT prior that conditions the
stage-2 refiner (classify-then-regress). Classification is field-robust, so we train on ALL centers
(341 AB+TH); the field-sensitive HU refiner trains on 0.35T only.

    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/train_sct_classifier.py \
        --out $WORKDIR/sct_runs/clf --epochs 200
"""
from __future__ import annotations

import os
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from train_sct_paired import norm_mr, load_arr   # reuse v4 MR norm (pct 1/99) + axis convention
import sct_aug   # approved sCT-front-end augmentation (3-axis flip + axial rot NEAREST-label + MR-only intensity)

DATA = (os.environ.get("WORKDIR", "./workdir") + "/sct_data_2mm.json")
# CT-HU band edges -> class id (air, lung, soft, bone); see sct_manifest classes
BANDS = [-700.0, -300.0, 200.0]   # <-700 air | -700..-300 lung | -300..200 soft | >200 bone
N_CLS = 4


def ct_to_class(ct_hu: np.ndarray) -> np.ndarray:
    return np.searchsorted(BANDS, ct_hu).astype(np.int64)   # band index -> class id (any #bands)


class ClsData(torch.utils.data.Dataset):
    def __init__(self, items, patch, train=True, aug=False, rot_deg=18.0):
        self.items = items; self.patch = patch; self.train = train
        self.aug = aug; self.rot_deg = rot_deg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        mr = norm_mr(load_arr(it["mr"])).astype(np.float32)
        lab = ct_to_class(load_arr(it["ct"]).astype(np.float32))
        if not self.train:
            return torch.from_numpy(mr)[None], torch.from_numpy(lab)
        ps = self.patch
        # pad if smaller than patch
        pad = [max(ps[d] - mr.shape[d], 0) for d in range(3)]
        if any(pad):
            pw = [(0, pad[d]) for d in range(3)]
            mr = np.pad(mr, pw); lab = np.pad(lab, pw)
        # foreground-biased crop: center on a bone or lung voxel when possible
        fg = np.argwhere((lab == 3) | (lab == 1))
        if len(fg) and np.random.rand() < 0.8:
            c = fg[np.random.randint(len(fg))]
            s = [int(np.clip(c[d] - ps[d] // 2, 0, mr.shape[d] - ps[d])) for d in range(3)]
        else:
            s = [np.random.randint(0, mr.shape[d] - ps[d] + 1) for d in range(3)]
        sl = tuple(slice(s[d], s[d] + ps[d]) for d in range(3))
        mr, lab = mr[sl], lab[sl]
        if self.aug:
            mr, lab = sct_aug.augment([mr, lab.astype(np.float32)], [False, True], 0,
                                      np.random, max_deg=self.rot_deg)
            lab = lab.astype(np.int64)
        elif np.random.rand() < 0.5:
            mr, lab = mr[::-1].copy(), lab[::-1].copy()
        return torch.from_numpy(mr)[None], torch.from_numpy(lab)


def model(n_cls=None):
    return UNet(spatial_dims=3, in_channels=1, out_channels=n_cls or N_CLS,
                channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
                num_res_units=2, norm="INSTANCE", act="LEAKYRELU")


def _pad16(n): return (16 - n % 16) % 16


@torch.no_grad()
def validate(net, items, patch, dev, whole=False):
    net.eval()
    dice = np.zeros(N_CLS); cnt = np.zeros(N_CLS)
    for it in items:
        mr = torch.from_numpy(norm_mr(load_arr(it["mr"])).astype(np.float32))[None, None].to(dev)
        gt = torch.from_numpy(ct_to_class(load_arr(it["ct"]).astype(np.float32))).to(dev)
        with torch.autocast("cuda"):
            if whole:                                  # whole-volume forward (matches whole-image training)
                Z, Y, X = mr.shape[-3:]
                xp = F.pad(mr, (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
                logits = net(xp)[..., :Z, :Y, :X]
            else:
                logits = sliding_window_inference(mr, patch, 4, net, overlap=0.25, mode="gaussian")
        pred = logits.argmax(1)[0]
        for c in range(N_CLS):
            p = pred == c; g = gt == c
            inter = (p & g).sum().float() * 2
            den = p.sum().float() + g.sum().float()
            if den > 0:
                dice[c] += float(inter / den); cnt[c] += 1
    net.train()
    return dice / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iters", type=int, default=140)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--bands", default=None, help="comma CT-HU band edges (e.g. -700,-300,200); "
                    "#classes = #bands+1. Default = the 4-class scheme.")
    ap.add_argument("--class-weights", default=None, help="comma CE weights per class")
    ap.add_argument("--whole-image", action="store_true", help="train+val on whole volumes (global "
                    "position context, esp. for lung) instead of 128^3 patches + sliding-window")
    ap.add_argument("--aug", action="store_true", help="enable approved sCT front-end augmentation "
                    "(3-axis flip + axial rotation NEAREST-for-label + MR-only intensity); default off "
                    "keeps the old single-axis flip for reproducibility")
    ap.add_argument("--rot-deg", type=float, default=18.0, help="max |axial rotation| when --aug")
    a = ap.parse_args()
    global BANDS, N_CLS
    if a.bands:
        BANDS = [float(x) for x in a.bands.split(",")]; N_CLS = len(BANDS) + 1
    dev = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = json.load(open(a.data)); patch = tuple(a.patch)
    if not a.whole_image:
        tr = ClsData(D["train"], patch, train=True, aug=a.aug, rot_deg=a.rot_deg)
        dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                         num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    else:
        rng = np.random.default_rng(0); tritems = D["train"]
    net = model(N_CLS).to(dev)
    # class weights: default = up-weight the rare/hard ends (4-class: air/lung/soft/bone)
    if a.class_weights:
        cw = torch.tensor([float(x) for x in a.class_weights.split(",")], device=dev)
    elif N_CLS == 4:
        cw = torch.tensor([0.5, 2.0, 1.0, 4.0], device=dev)
    else:
        cw = torch.ones(N_CLS, device=dev)
    assert len(cw) == N_CLS, f"class-weights {len(cw)} != N_CLS {N_CLS}"
    lossf = DiceCELoss(to_onehot_y=True, softmax=True, weight=cw)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best = -1.0; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); run = 0.0
        if not a.whole_image:
            it = iter(dl)
        for step in range(a.iters):
            if a.whole_image:                          # one whole volume per step (batch 1, pad /16)
                d = tritems[rng.integers(len(tritems))]
                mr = norm_mr(load_arr(d["mr"])).astype(np.float32)
                lab = ct_to_class(load_arr(d["ct"]).astype(np.float32))
                if a.aug:
                    mr, lab = sct_aug.augment([mr, lab.astype(np.float32)], [False, True], 0,
                                              rng, max_deg=a.rot_deg)
                    lab = lab.astype(np.int64)
                elif rng.random() < 0.5:
                    mr, lab = mr[::-1].copy(), lab[::-1].copy()
                x = torch.from_numpy(mr)[None, None].to(dev); y = torch.from_numpy(lab)[None, None].to(dev)
                Z, Y, X = x.shape[-3:]; x = F.pad(x, (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
                y = F.pad(y, (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)), value=0)
            else:
                try: x, y = next(it)
                except StopIteration: it = iter(dl); x, y = next(it)
                x = x.to(dev); y = y[:, None].to(dev)
            with torch.autocast("cuda"):
                loss = lossf(net(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); run += loss.item()
            if a.smoke and step + 1 >= a.smoke:
                print(f"[smoke] ep{ep} step{step+1} loss {loss.item():.4f}", flush=True)
                if ep >= 1: return
        sched.step()
        if ep % 5 == 0 or ep == 1:
            d = validate(net, D["val"], patch, dev, whole=a.whole_image)
            mean_fb = float(d[1:].mean())   # mean Dice over non-air classes
            tag = ""
            if mean_fb > best:
                best = mean_fb; torch.save({"net": net.state_dict(), "ep": ep}, out / "best.pt"); tag = " *best*"
            torch.save({"net": net.state_dict(), "ep": ep}, out / "last.pt")
            dstr = " ".join(f"c{c}:{d[c]:.3f}" for c in range(N_CLS))
            print(f"ep {ep}/{a.epochs} | loss {run/a.iters:.4f} | Dice {dstr} | "
                  f"{(time.time()-t0)/60:.1f}min{tag}", flush=True)


if __name__ == "__main__":
    main()
