"""Lever-4 stage 2: whole-image MR + coarse-CT -> fine sCT (classify-then-regress refiner).

Input = [normalised MR, normalised coarse CT (from the tissue classifier)]. The refiner learns the fine
HU *residual* on top of a correctly-located coarse tissue map, instead of hallucinating bone/lung
density from MR alone. Whole-image -> whole-image (train AND infer the same way; avoids the INSTANCE-norm
train/test mismatch that wrecked sliding-window inference of full-volume-trained nets). Region-weighted
L1 (bone & lung up-weighted) folds in lever 3. Trained on 0.35T (Center-B) only -> field-matched HU.

    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/train_sct_refiner.py \
        --out $WORKDIR/sct_runs/refiner --coarse-dir $WORKDIR/cache/coarse_ct

PERF: the per-step cost is single-threaded CPU (load-once cache + full-volume augmentation), so with the
inline loop the GPU sits ~90% idle waiting for the CPU. Pass --workers N to move sampling+aug into N
DataLoader worker processes that overlap with GPU compute. The volume cache is pre-populated in the main
process BEFORE the workers fork, so on Linux (fork start method) the big decoded arrays are shared
copy-on-write — workers only READ them (aug returns fresh copies), so RAM is not duplicated N times.
--workers 0 (default) keeps the exact original inline behaviour.
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
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from train_sct_paired import norm_mr, load_arr
import sct_aug   # approved sCT-front-end augmentation (all 3 vols continuous -> LINEAR; MR-only intensity)

DATA = (os.environ.get("WORKDIR", "./workdir") + "/sct_data_2mm.json")
_PATCH = None
CT_LO, CT_HI = -1000.0, 2000.0
_T_BODY = (-500.0 - CT_LO) / (CT_HI - CT_LO)
_T_LUNG = (-300.0 - CT_LO) / (CT_HI - CT_LO)
_T_BONE = (200.0 - CT_LO) / (CT_HI - CT_LO)


def to01(hu):
    return np.clip((hu - CT_LO) / (CT_HI - CT_LO), 0.0, 1.0).astype(np.float32)


def _pad16(n): return (16 - n % 16) % 16


_NORM = "INSTANCE"   # set to ("group", {"num_groups": 8}) via --norm group for patch-invariant training
def model():
    return UNet(spatial_dims=3, in_channels=2, out_channels=1,
                channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
                num_res_units=2, norm=_NORM, act="LEAKYRELU")


_CACHE = {}   # pid -> (mr, coarse01, ct01) f32; load+decompress each native volume ONCE (was 1.7s/iter).
_BONE = {}    # pid -> argwhere(ct01>_T_BONE) coords, computed once (was 92 ms/iter).
# Safe to return cached refs without copying: the patch loop crops a window (view) then augments a fresh
# array, and validate() only reads. Do NOT mutate these cached arrays in place. With --workers, this dict
# is populated in the main process and inherited copy-on-write by the fork'd workers (read-only there).
def load_case(it, coarse_dir):
    hit = _CACHE.get(it["pid"])
    if hit is not None:
        return hit
    mr = norm_mr(load_arr(it["mr"])).astype(np.float32)
    ct01 = to01(load_arr(it["ct"]).astype(np.float32))
    coarse01 = to01(load_arr(os.path.join(coarse_dir, it["pid"] + ".nii.gz")).astype(np.float32))
    _CACHE[it["pid"]] = (mr, coarse01, ct01)
    return mr, coarse01, ct01

def _bone_coords(pid, ct01):
    hit = _BONE.get(pid)
    if hit is None:
        hit = np.argwhere(ct01 > _T_BONE)
        _BONE[pid] = hit
    return hit


def make_sample(train, coarse_dir, patch, aug, rot_deg, rng):
    """One training sample: pick a case, load (cached), and either crop+aug a patch or aug the whole
    volume. Returns (mr, coarse01, ct01). Identical to the original inline body; shared by the inline
    loop (--workers 0) and the DataLoader path so behaviour cannot diverge."""
    it = train[rng.integers(len(train))]
    mr, coarse01, ct01 = load_case(it, coarse_dir)
    if patch is not None:
        # EFFICIENT patch path. sct_aug rotation is in the (y,z) plane (axes 1,2) so only y needs a
        # crop margin; x is not rotated and z is kept in full (native z=94 < patch). We therefore
        # aug a small xy-margin window at FULL z (192x192x94, ~232ms) instead of a z-padded 192^3
        # (~547ms) or the whole native volume (~3s). center-crop xy -> patch, then pad z -> patch.
        ps = patch; MARG = 32
        wx, wy = ps[0] + 2*MARG, ps[1] + 2*MARG
        fg = _bone_coords(it["pid"], ct01)                 # cached bone voxels
        if len(fg) and rng.random() < 0.8:
            c = fg[rng.integers(len(fg))]
        else:
            c = [rng.integers(0, s) for s in ct01.shape]
        lox = int(np.clip(c[0]-wx//2, 0, max(mr.shape[0]-wx, 0)))
        loy = int(np.clip(c[1]-wy//2, 0, max(mr.shape[1]-wy, 0)))
        sl = (slice(lox, min(lox+wx, mr.shape[0])), slice(loy, min(loy+wy, mr.shape[1])), slice(None))
        mw, cw, ctw = mr[sl], coarse01[sl], ct01[sl]       # xy window, full z (views)
        pxy = [(0, max(wx-mw.shape[0],0)), (0, max(wy-mw.shape[1],0)), (0,0)]
        if pxy[0][1] or pxy[1][1]:
            mw=np.pad(mw,pxy); cw=np.pad(cw,pxy); ctw=np.pad(ctw,pxy)
        if aug:
            mw, cw, ctw = sct_aug.augment([mw, cw, ctw], [False, False, False], 0, rng, max_deg=rot_deg)
        elif rng.random() < 0.5:
            mw, cw, ctw = mw[::-1].copy(), cw[::-1].copy(), ctw[::-1].copy()
        csx, csy = (mw.shape[0]-ps[0])//2, (mw.shape[1]-ps[1])//2  # center-crop xy -> patch
        mw = mw[csx:csx+ps[0], csy:csy+ps[1]]; cw = cw[csx:csx+ps[0], csy:csy+ps[1]]; ctw = ctw[csx:csx+ps[0], csy:csy+ps[1]]
        zp = ps[2]-mw.shape[2]                             # z: pad up (native z<patch) or random-crop
        if zp > 0:
            zpw=[(0,0),(0,0),(0,zp)]; mw=np.pad(mw,zpw); cw=np.pad(cw,zpw); ctw=np.pad(ctw,zpw)
        elif zp < 0:
            zst=int(rng.integers(0,mw.shape[2]-ps[2]+1)); mw=mw[:,:,zst:zst+ps[2]]; cw=cw[:,:,zst:zst+ps[2]]; ctw=ctw[:,:,zst:zst+ps[2]]
        mr, coarse01, ct01 = mw, cw, ctw
    else:                                                 # whole-image path: aug full volume
        if aug:
            mr, coarse01, ct01 = sct_aug.augment([mr, coarse01, ct01], [False, False, False], 0,
                                                 rng, max_deg=rot_deg)
        elif rng.random() < 0.5:
            mr, coarse01, ct01 = mr[::-1].copy(), coarse01[::-1].copy(), ct01[::-1].copy()
    return mr, coarse01, ct01


class RefinerDataset(Dataset):
    """Length = iters; each __getitem__ draws one random augmented sample. rng is replaced per worker
    (worker_init_fn) so the workers don't all draw the same stream."""
    def __init__(self, train, coarse_dir, patch, aug, rot_deg, iters):
        self.train, self.coarse_dir, self.patch = train, coarse_dir, patch
        self.aug, self.rot_deg, self.n = aug, rot_deg, int(iters)
        self.rng = np.random.default_rng(0)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        mr, c, ct = make_sample(self.train, self.coarse_dir, self.patch, self.aug, self.rot_deg, self.rng)
        x = torch.from_numpy(np.ascontiguousarray(np.stack([mr, c], 0)))   # (2,Z,Y,X)
        y = torch.from_numpy(np.ascontiguousarray(ct[None]))               # (1,Z,Y,X)
        return x, y

def _winit(wid):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(12345 + wid)

def _collate(batch):     # batch_size=1: unwrap and add the batch dim main expects
    x, y = batch[0]
    return x.unsqueeze(0), y.unsqueeze(0)                                   # (1,2,Z,Y,X), (1,1,Z,Y,X)


def train_step(net, x, y, opt, scaler, a):
    """Pad to /16, forward, region-weighted L1 (+ optional grad-L1), backward. x,y already on device."""
    Z, Y, X = x.shape[-3:]
    x = F.pad(x, (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
    y = F.pad(y, (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
    with torch.autocast("cuda"):
        po = net(x)[:, :, :Z, :Y, :X]; yo = y[:, :, :Z, :Y, :X]
        loss = wloss(po, yo, a.w_bone, a.w_lung) + (a.grad_w * grad_l1(po, yo) if a.grad_w > 0 else 0.0)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    return loss.item()


def wloss(pred, ct01, wb, wl):
    err = (pred - ct01).abs()
    body = ct01 > _T_BODY; bone = ct01 > _T_BONE; lung = body & (ct01 < _T_LUNG)
    w = torch.ones_like(ct01) + (wb - 1.0) * bone.float() + (wl - 1.0) * lung.float()
    return (err * w).sum() / w.sum()

def grad_l1(pred, ct01):
    """L1 on finite-difference gradients along z,y,x — matches EDGE LOCATION/magnitude (Exp: 52% of sCT
    error is at tissue boundaries). Nudges the refiner to place bone/lung interfaces where the real CT
    has them, without forcing over-sharp edges (a small weight)."""
    l = 0.0
    for d in (2, 3, 4):
        gp = pred.diff(dim=d); gc = ct01.diff(dim=d)
        l = l + (gp - gc).abs().mean()
    return l / 3.0


@torch.no_grad()
def validate(net, items, coarse_dir, dev):
    net.eval(); hu = {"all": [], "bone": [], "lung": []}
    for it in items:
        mr, coarse01, ct01 = load_case(it, coarse_dir)
        x = np.stack([mr, coarse01], 0)[None]
        Z, Y, X = x.shape[-3:]
        xt = F.pad(torch.from_numpy(x).to(dev), (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
        with torch.autocast("cuda"):
            if _PATCH is not None:
                p = sliding_window_inference(xt, _PATCH, 2, net, overlap=0.25, mode="gaussian")[0,0,:Z,:Y,:X].float().cpu().numpy()
            else:
                p = net(xt)[0, 0, :Z, :Y, :X].float().cpu().numpy()
        ct = ct01 * (CT_HI - CT_LO) + CT_LO; ps = np.clip(p, 0, 1) * (CT_HI - CT_LO) + CT_LO
        body = ct > -500; e = np.abs(ps - ct)
        hu["all"].append(e[body].mean())
        if (ct > 200).any(): hu["bone"].append(e[(ct > 200) & body].mean())
        if ((ct < -300) & body).any(): hu["lung"].append(e[(ct < -300) & body].mean())
    net.train()
    return {k: float(np.mean(v)) for k, v in hu.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--coarse-dir", required=True)
    ap.add_argument("--data", default=DATA); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--iters", type=int, default=166); ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--w-bone", type=float, default=6.0); ap.add_argument("--w-lung", type=float, default=3.0)
    ap.add_argument("--grad-w", type=float, default=0.0, help="weight on edge-location gradient-L1 loss (0=off)")
    ap.add_argument("--norm", choices=["instance", "group"], default="instance", help="group = patch-invariant (fixes whole-vs-patch INSTANCE mismatch)")
    ap.add_argument("--init-from", default=""); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--aug", action="store_true", help="enable approved sCT front-end augmentation "
                    "(3-axis flip + axial rotation LINEAR + MR-only intensity); default off keeps the "
                    "old single-axis flip for reproducibility")
    ap.add_argument("--rot-deg", type=float, default=18.0, help="max |axial rotation| when --aug")
    ap.add_argument("--patch", type=int, nargs=3, default=None, help="patch-based train (for 1x1x3 whole-image OOM); sliding-window validate")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes overlapping "
                    "aug with GPU compute (0 = original inline loop). Cache is pre-populated before fork "
                    "so decoded volumes are shared copy-on-write, not duplicated per worker.")
    ap.add_argument("--allfield", action="store_true", help="train on ALL last-year MRI (every field "
                    "strength, 341) instead of 0.35T-only; up-weight same-field via --samefield_weight")
    ap.add_argument("--samefield_weight", type=int, default=2, help="duplicate 0.35T (same-field) entries "
                    "this many times when --allfield (2 = same-field seen 2x as often as other-field)")
    a = ap.parse_args(); dev = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global _PATCH; _PATCH = a.patch
    global _NORM; _NORM = ("group", {"num_groups": 8}) if a.norm == "group" else "INSTANCE"
    D = json.load(open(a.data))
    if a.allfield:
        # user idea 2026-08-19: use ALL last-year MRI (every field strength) but UP-WEIGHT the same-field
        # (0.35T, = this year's competition) by duplicating those entries a.samefield_weight times. Uniform
        # sampling over the duplicated list => effective per-sample weight, works for BOTH the inline loop
        # and the DataLoader path (no custom sampler needed).
        base = list(D["train"])                                  # all 341 (0.35T + high)
        sf = [it for it in base if it.get("field") == "0.35T"]
        train = base + sf * max(a.samefield_weight - 1, 0)
        nsf = len(sf); noth = len(base) - nsf
        print(f"refiner train (ALL-field, samefield x{a.samefield_weight}): {len(train)} "
              f"= {noth} high + {nsf}x{a.samefield_weight} same-field(0.35T) | val: {len(D['val'])}")
    else:
        refp = set(D.get("refiner_pids", [it["pid"] for it in D["train"]]))
        train = [it for it in D["train"] if it["pid"] in refp]   # 0.35T only (original)
        print(f"refiner train (0.35T only): {len(train)} | val: {len(D['val'])}")
    net = model().to(dev)
    if a.init_from:
        sd = torch.load(a.init_from, map_location=dev); net.load_state_dict(sd.get("net", sd), strict=False)
        print(f"warm-start <- {a.init_from}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler(); best = 1e9; t0 = time.time(); rng = np.random.default_rng(0)

    loader = None
    if a.workers > 0:
        # Pre-populate the decoded-volume cache (and bone coords for patch mode) in THIS process so the
        # fork'd workers inherit them copy-on-write instead of each re-decoding + duplicating RAM.
        torch.multiprocessing.set_sharing_strategy("file_system")   # avoid fd exhaustion with many tensors
        print(f"pre-caching {len(train)} volumes for {a.workers} workers ...", flush=True)
        for it in train:
            _, _, ct01 = load_case(it, a.coarse_dir)
            if a.patch is not None:
                _bone_coords(it["pid"], ct01)
        ds = RefinerDataset(train, a.coarse_dir, a.patch, a.aug, a.rot_deg, a.iters)
        loader = DataLoader(ds, batch_size=1, num_workers=a.workers, persistent_workers=True,
                            prefetch_factor=2, collate_fn=_collate, worker_init_fn=_winit, pin_memory=True)

    for ep in range(1, a.epochs + 1):
        net.train(); run = 0.0
        if loader is not None:
            step = 0
            for x, y in loader:
                run += train_step(net, x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), opt, scaler, a)
                step += 1
                if a.smoke and step >= a.smoke:
                    print(f"[smoke] ep{ep} step{step} peak {torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
                    return
        else:
            for step in range(a.iters):
                mr, coarse01, ct01 = make_sample(train, a.coarse_dir, a.patch, a.aug, a.rot_deg, rng)
                x = torch.from_numpy(np.stack([mr, coarse01], 0)[None]).to(dev)
                y = torch.from_numpy(ct01[None, None]).to(dev)
                run += train_step(net, x, y, opt, scaler, a)
                if a.smoke and step + 1 >= a.smoke:
                    print(f"[smoke] ep{ep} step{step+1} peak {torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
                    return
        sched.step()
        if ep % 5 == 0 or ep == 1:
            v = validate(net, D["val"], a.coarse_dir, dev); tag = ""
            if v["all"] < best:
                best = v["all"]; torch.save({"net": net.state_dict(), "ep": ep}, out / "best.pt"); tag = " *best*"
            torch.save({"net": net.state_dict(), "ep": ep}, out / "last.pt")
            print(f"ep {ep}/{a.epochs} | loss {run/a.iters:.4f} | HU-MAE all {v['all']:.1f} "
                  f"bone {v['bone']:.1f} lung {v['lung']:.1f} | {(time.time()-t0)/60:.1f}min{tag}", flush=True)


if __name__ == "__main__":
    main()
