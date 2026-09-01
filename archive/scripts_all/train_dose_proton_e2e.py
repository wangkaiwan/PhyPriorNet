"""Task-4 (Proton-MRI) end-to-end dose-aware training: MRI -> wholesoft sCT -> proton dose.

The proton analogue of train_dose_e2e (Fuse-A14): the sCT synth and the proton dose net are jointly
trained; the proton dose loss back-props THROUGH the differentiable proton physics (WEPL ray-march +
GPU Hong PB prior, both differentiable w.r.t. density) into the synth -> the sCT becomes proton-dose-aware.

Resolution: the synth was trained at 2mm (photon grid); proton dose/channels/GT are at native 1x1x3mm.
So per step: synth makes sCT at 2mm, a DIFFERENTIABLE physical-coordinate grid_sample resamples it to the
native grid, and the proton physics runs natively. dose gradient flows native -> resample -> 2mm synth.

Warm-start: synth = wholesoft fusion synth (Task-2 SOTA 90.7); dose = UNMASKED proton champion
(GPU-PB ft + WEPL-fix, 97.0; masking was rejected, see task3). Conservative LR finetune.

Custom in-process loop (no DataLoader): per step one patient -> one sCT(2mm)->native density (amortised
over k_cps beamlets) -> per beamlet sample a patch, compute diff channels on the patch bbox, dose net, loss.
"""
from __future__ import annotations
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # train_dose_e2e import
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from doserad.physics.diff_channels import hu_to_density_torch
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine

DENS_MAX = 2.5   # density-direct: synth out [0,1] -> density [0, DENS_MAX] g/cm^3 (covers cortical bone ~1.9)
from doserad.physics.proton_pb_gpu import (ProtonMachineData, proton_pb_dose_gpu,
                                           _wepl_crop, _compute_ssd, _SSD_DENSITY_THRESHOLD)
from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry
from doserad.losses.dose_loss import weighted_l1
from doserad.data.proton_dataset import (PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR,
                                          ProtonDoseDataset, build_proton_rows)
from torch.utils.data import DataLoader
from doserad.io.mha import load_mha
from doserad.train.loop import build_optim, _ema_update, _save, _init_wandb, _wlog
from train_dose_e2e import E2E, CT_LO, CT_HI, _pad16, sct_anchor_loss

PROTON_ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"   # 2mm MR/CT the synth was trained on
_PCS = torch.tensor(_P_CH_SCALE_PRIOR, dtype=torch.float32).view(-1, 1, 1, 1)


def _resample_grid(src_geom, dst_geom, dev):
    """Constant grid mapping each NATIVE (dst) voxel center -> normalized coord in the 2mm (src) volume,
    via shared world coordinates. For F.grid_sample(src(1,1,Zs,Ys,Xs), grid)->(1,1,Zd,Yd,Xd)."""
    (sx, sy, sz), (ox, oy, oz), (Zs, Ys, Xs) = src_geom
    (dx, dy, dz), (dox, doy, doz), (Zd, Yd, Xd) = dst_geom
    xs = dox + torch.arange(Xd, device=dev, dtype=torch.float32) * dx
    ys = doy + torch.arange(Yd, device=dev, dtype=torch.float32) * dy
    zs = doz + torch.arange(Zd, device=dev, dtype=torch.float32) * dz
    gx = (xs - ox) / max(sx * (Xs - 1), 1e-6) * 2 - 1
    gy = (ys - oy) / max(sy * (Ys - 1), 1e-6) * 2 - 1
    gz = (zs - oz) / max(sz * (Zs - 1), 1e-6) * 2 - 1
    GZ, GY, GX = torch.meshgrid(gz, gy, gx, indexing="ij")
    return torch.stack([GX, GY, GZ], dim=-1).unsqueeze(0)        # (1,Zd,Yd,Xd,3)


def _wepl_tensor(density_t, spacing, origin, src, bb, sad, dev):
    """Differentiable WEPL (g/cm^2) on patch bbox bb, computed on the (torch) native density."""
    z0, z1, y0, y1, x0, x1 = bb
    ox, oy, oz = origin; sx, sy, sz = spacing
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)
    axis = np.asarray(src["axis"], np.float32)
    ssd = _compute_ssd(density_t.detach(), spacing, origin, src["src"], axis, sad, dev,
                       threshold=_SSD_DENSITY_THRESHOLD)
    return _wepl_crop(density_t, spacing, origin, src["src"], coords, dev,
                      march_start_mm=max(ssd - 50.0, 0.0))        # (dz,dy,dx) tensor, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="resume model/opt/sched/step from <run_dir>/state.pt (for quiet-hour kills)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda"
    # skin-entry PB prior (per-ray entered gate, == CT proton path). Opt-in via cfg; the WEPL
    # channel is unchanged (air contributes ~0 to path length, same as CT proton weplfix).
    skin_entry = bool(cfg.get("skin_entry", False))
    pb_fn = proton_pb_dose_gpu_skinentry if skin_entry else proton_pb_dose_gpu
    pm = ProtonMachineData(device=dev)
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    anchors = machine.hu_anchors
    run_dir = Path(cfg["run_root"]) / cfg["exp_name"]; run_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["cache_dir"])
    coarse_dir = cfg["coarse_dir"]
    max_steps = a.max_steps or cfg["max_steps"]
    k_cps = 1 if a.smoke else int(cfg.get("k_cps", 2))
    P = tuple(cfg.get("patch", [32, 128, 128]))
    fg_prob = float(cfg.get("fg_prob", 0.7))
    lam = float(cfg.get("lam_sct", 0.5))
    hi_frac, hi_w = float(cfg["hi_frac"]), float(cfg["hi_w"])
    grad_w = float(cfg.get("grad_w", 0.0)); het_w = float(cfg.get("het_w", 0.0))
    lung_w = float(cfg.get("lung_w", 0.0)); lung_lo = float(cfg.get("lung_lo", 0.02))
    lung_hi = float(cfg.get("lung_hi", 0.6))
    sct_wb = float(cfg.get("sct_w_bone", 1.0))      # uncertainty insight: up-weight recoverable bone/soft
    sct_wl = float(cfg.get("sct_w_lung", 1.0))      # down-weight unrecoverable lung density (MR-limited)
    wepl_w = float(cfg.get("wepl_w", 0.0))          # range-consistency: L1(sCT-WEPL, realCT-WEPL) — proton-specific
    wepl_dir = cfg.get("wepl_dir")                   # weplfix cache = real-CT WEPL target
    density_direct = bool(cfg.get("density_direct", False))  # synth outputs density directly (skip HU bottleneck)
    # sct_anchor_loss' default thresholds are in HU-01 space, but with density_direct the anchor is
    # dens01 = density/DENS_MAX. Feeding HU-01 thresholds to density data puts "bone" at 0.4*2.5 =
    # 1.0 g/cm^3 -- water -- so the bone mask would swallow every soft tissue and the weighting would
    # degenerate into weighting the whole body. Map the SAME HU thresholds through hu_to_density
    # instead (bone lands at 1.148 g/cm^3). Mirrors train_dose_e2e.py:115; None keeps the HU-01
    # defaults when the anchor really is ct01.
    _dthr = tuple(float(hu_to_density(np.array([h], np.float32), anchors)[0] / DENS_MAX)
                  for h in (-500.0, -300.0, 200.0)) if density_direct else None
    global _PCS; _PCS = _PCS.to(dev)

    splits = json.load(open(cfg["splits"])); fold = splits[f"fold_{cfg['fold']}"]
    train_pts = [p for p in fold["train"] if (cache / p).is_dir()]
    val_pts = [p for p in fold["val"] if (cache / p).is_dir()]
    bfiles = {p: sorted(f.name for f in (cache / p).glob("B*_R*_L*.npz")) for p in train_pts + val_pts}
    rng = random.Random(0)

    plans = {}
    def rays_of(pid):
        if pid not in plans:
            pl = json.load(open(Path(PROTON_ROOT) / pid / f"{pid}.json"))
            d = {}
            for b in pl["beams"]:
                for r in b["rays"]:
                    for bl in r["beamlets"]:
                        tgt = np.asarray(r["ray_target"], np.float64); js = np.asarray(r["ray_source"], np.float64)
                        ax = tgt - js; ax = ax / (np.linalg.norm(ax) + 1e-12)
                        srcp = (tgt - ax * pm.sad).astype(np.float32)
                        d[(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"])] = {
                            "src": srcp, "axis": ax.astype(np.float32),
                            "ray_source": r["ray_source"], "ray_target": r["ray_target"], "e": bl["energy"]}
            plans[pid] = d
        return plans[pid]

    vol_cache = {}
    _SYNTH_ROOT = PROTON_ROOT if bool(cfg.get("native_synth", False)) else PHOTON_ROOT  # native=1x1x3 synth
    def load_vol(pid):
        if pid not in vol_cache:
            import SimpleITK as sitk
            mr = sitk.ReadImage(f"{_SYNTH_ROOT}/{pid}/image/mr.mha")            # 2mm (PHOTON) or native (PROTON)
            a_mr = sitk.GetArrayFromImage(mr).astype(np.float32)
            lo, hi = np.percentile(a_mr, 1), np.percentile(a_mr, 99)
            mr01 = np.clip((a_mr - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)
            cv = load_mha(Path(coarse_dir) / f"{pid}.nii.gz").array.astype(np.float32)
            coarse01 = np.clip((cv - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)
            ct2 = sitk.GetArrayFromImage(sitk.ReadImage(f"{_SYNTH_ROOT}/{pid}/image/ct.mha")).astype(np.float32)
            ct01 = None if density_direct else np.clip((ct2 - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)
            dens01 = np.clip(hu_to_density(ct2, anchors) / DENS_MAX, 0, 1).astype(np.float32)  # density-direct anchor
            ctn = load_mha(Path(PROTON_ROOT) / pid / "image" / "ct.mha")        # native geometry
            src_geom = (mr.GetSpacing(), mr.GetOrigin(), tuple(mr.GetSize()[::-1]))
            dst_geom = (ctn.spacing, ctn.origin, ctn.array.shape)
            # grid is REBUILT on-GPU per step in density_native (deterministic, ~30ms) so we can cache MANY
            # vols (cfg.vol_cache_n) without holding N grids on GPU -> would OOM. Cost: 3 CPU arrays/vol only.
            vol_cache[pid] = dict(mr01=mr01, coarse01=coarse01, ct01=ct01, dens01=dens01,
                                  spacing=ctn.spacing, origin=ctn.origin, shape=ctn.array.shape,
                                  src_geom=src_geom, dst_geom=dst_geom)
            if len(vol_cache) > int(cfg.get("vol_cache_n", 4)):
                vol_cache.pop(next(iter(vol_cache)))
        return vol_cache[pid]

    net = E2E(cfg).to(dev)
    ws = torch.load(cfg["synth_ckpt"], map_location=dev); wss = ws.get("ema", ws.get("model"))
    net.synth.load_state_dict({k[len("synth."):]: v for k, v in wss.items() if k.startswith("synth.")})
    ds = torch.load(cfg["dose_ckpt"], map_location=dev); net.dose.load_state_dict(ds.get("ema", ds.get("model")))
    print(f"[warm-start] synth<-{cfg['synth_ckpt']}  dose<-{cfg['dose_ckpt']}", flush=True)
    if cfg.get("init_from"):   # warm-start the FULL E2E (synth+dose) from a prior proton-MRI checkpoint
        ist = torch.load(cfg["init_from"], map_location=dev)
        net.load_state_dict(ist.get("ema", ist.get("model")))
        print(f"[init] full E2E <- {cfg['init_from']}", flush=True)
    if bool(cfg.get("freeze_synth", False)):
        for p in net.synth.parameters():
            p.requires_grad_(False)
        print("[freeze] synth frozen (dose-net + CT-mix only)", flush=True)
    opt, sched, scaler, ema = build_optim(net, max_steps, lr=cfg["lr"], wd=cfg["weight_decay"])
    mod = torch.zeros(1, dtype=torch.long, device=dev)
    run = _init_wandb(cfg, run_dir) if cfg.get("wandb") and not a.smoke else None

    mr_aug_cfg = cfg.get("mr_aug", None)   # e.g. {bias:0.5, gamma:[0.7,1.6], noise:0.03} — MR-intensity
    # aug on the SYNTH input ONLY (not the GT dose/geometry) to make the sCT robust to cross-scanner
    # intensity/contrast shift (the diagnosed cause of the MRI internal→leaderboard gap). Applied per
    # training step so the synth sees varied intensity; validation/reference calls pass aug=False.

    def density_native(V, aug=False):
        """sCT (2mm) -> native density (torch, differentiable through synth)."""
        mri = torch.from_numpy(V["mr01"]).to(dev)[None, None]
        if aug and mr_aug_cfg:
            Z, Y, X = mri.shape[-3:]
            amp = float(np.random.uniform(0.0, float(mr_aug_cfg.get("bias", 0.5))))
            small = torch.randn(1, 1, max(Z // 8, 2), max(Y // 8, 2), max(X // 8, 2), device=dev)
            b = F.interpolate(small, size=(Z, Y, X), mode="trilinear", align_corners=False)
            b = 1.0 + amp * (b / (b.abs().amax() + 1e-6))
            mri = mri * b
            mri = (mri / (mri.amax() + 1e-6)).clamp(0, 1)                       # renorm (survives pct-norm)
            gl, gh = mr_aug_cfg.get("gamma", [0.7, 1.6])
            mri = mri.clamp(0, None) ** float(np.random.uniform(gl, gh))        # contrast
            sn = float(mr_aug_cfg.get("noise", 0.03))
            if sn > 0:
                mri = (mri + float(np.random.uniform(0, sn)) * torch.randn_like(mri)).clamp(0, 1)
        co = torch.from_numpy(V["coarse01"]).to(dev)[None, None]
        sct01 = net.sct01(torch.cat([mri, co], 1))[0, 0]            # 2mm [0,1]
        if density_direct:                                          # synth out -> density (clamped; UNet unbounded)
            dens2 = sct01.clamp(0, 1) * DENS_MAX
        else:
            dens2 = hu_to_density_torch(sct01 * (CT_HI - CT_LO) + CT_LO, anchors)   # 2mm g/cm^3
        grid = _resample_grid(V["src_geom"], V["dst_geom"], dev)   # rebuilt per step (cache holds geoms, not grid)
        densn = F.grid_sample(dens2[None, None], grid, mode="bilinear",
                              align_corners=True, padding_mode="border")[0, 0]   # native
        return sct01, densn

    def _patch(dz, dy, dx, fg):
        pz, py, px = min(P[0], dz), min(P[1], dy), min(P[2], dx)
        if len(fg):
            c = fg[rng.randrange(len(fg))]
            s0 = int(np.clip(c[0] - pz // 2, 0, dz - pz)); s1 = int(np.clip(c[1] - py // 2, 0, dy - py))
            s2 = int(np.clip(c[2] - px // 2, 0, dx - px))
        else:
            s0, s1, s2 = (dz - pz) // 2, (dy - py) // 2, (dx - px) // 2
        return s0, s1, s2, pz, py, px

    @torch.no_grad()
    def validate():
        net.eval(); tot = 0.0; n = 0
        for pid in val_pts[:6]:
            V = load_vol(pid); _, densn = density_native(V); rs = rays_of(pid)
            for name in bfiles[pid][:4]:
                z = np.load(cache / pid / name); ch = z["channels"].astype(np.float32)
                bb = tuple(int(v) for v in z["bbox"]); dose_np = z["dose"].astype(np.float32)
                key = tuple(int(name[:-4].split("_")[i][1:]) for i in range(3))
                pred = _beamlet_pred(densn, ch, bb, rs[key], V) / PROTON_DOSE_SCALE
                gt = torch.from_numpy(dose_np).to(dev)
                m = gt >= 0.1 * gt.max()
                if m.any():
                    tot += float((pred[m] - gt[m]).abs().mean() / gt[m].mean().clamp_min(1e-9)) * 100; n += 1
        net.train()
        return tot / max(n, 1)

    def _beamlet_pred(densn, ch, bb, ray, V, sl_full=None):
        """differentiable proton dose for the full beamlet bbox (used in val)."""
        z0, z1, y0, y1, x0, x1 = bb
        dens_c = densn[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        wepl = _wepl_tensor(densn, V["spacing"], V["origin"], ray, bb, pm.sad, dev)
        pb = pb_fn(_Img(V), ray["ray_source"], ray["ray_target"], ray["e"], out_bbox=bb,
                   machine=pm, density_override=densn, device=dev, return_tensor=True)
        cht = torch.from_numpy(ch).to(dev)
        inp = torch.stack([dens_c, wepl, pb * PROTON_DOSE_SCALE, cht[2], cht[3]], 0) / _PCS
        Z, Y, X = inp.shape[-3:]
        inp = F.pad(inp[None], (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
        return net.dose(inp, mod)[0, 0, :Z, :Y, :X].float()

    class _Img:   # lightweight image-geometry shim for proton_pb_dose_gpu (needs .array.shape/.spacing/.origin)
        def __init__(self, V): self.array = np.empty(V["shape"], np.float32); self.spacing = V["spacing"]; self.origin = V["origin"]

    # ---- multi-modal CT-mix: interleave real-CT proton beamlet batches (dose net only) ----
    # Uses the SAME ProtonDoseDataset as real proton-CT training (proton_ssd channels + skin-entry PB prior
    # + weplfix WEPL), so a CT step is byte-identical to dedicated proton-CT training. No synth, no diff
    # physics on these steps. Opt-in via cfg.ct_mix_cache (default off = unchanged e2e).
    ct_iter = None; ct_mix_prob = float(cfg.get("ct_mix_prob", 0.5))
    if cfg.get("ct_mix_cache"):
        ctc = cfg["ct_mix_cache"]
        ct_rows = [r for r in build_proton_rows(ctc, train_pts)]
        ct_ds = ProtonDoseDataset(ct_rows, ctc, prior_dir=cfg.get("ct_mix_prior_dir"),
                                  wepl_dir=cfg.get("ct_mix_wepl_dir"), patch=P, fg_prob=fg_prob)
        ct_loader = DataLoader(ct_ds, batch_size=int(cfg.get("ct_mix_batch", 1)), shuffle=True,
                               num_workers=int(cfg.get("ct_num_workers", 2)), drop_last=True)
        def _ct_gen():
            while True:
                for b in ct_loader:
                    yield b
        ct_iter = iter(_ct_gen())
        print(f"[ct-mix] {len(ct_rows)} CT proton beamlets from {ctc}, prob {ct_mix_prob} (dose-net only)", flush=True)

    start_step = 0
    if a.resume and (run_dir / "state.pt").exists():
        rst = torch.load(run_dir / "state.pt", map_location=dev)
        net.load_state_dict(rst["model"]); ema.load_state_dict(rst["ema"])
        opt.load_state_dict(rst["opt"]); sched.load_state_dict(rst["sched"])
        scaler.load_state_dict(rst["scaler"]); start_step = int(rst["step"])
        print(f"[resume] from step {start_step} ({run_dir}/state.pt)", flush=True)

    best = float("inf"); t0 = time.time()
    _PROF = os.environ.get("DOSERAD_PROFILE") == "1"   # per-step load-vs-compute timing (profiling only)
    for step in range(start_step + 1, max_steps + 1):
        if ct_iter is not None and rng.random() < ct_mix_prob:
            bb = next(ct_iter)
            ci = bb["input"].to(dev); cd = bb["dose"].to(dev); cdens = bb["density"].to(dev)
            with torch.autocast("cuda"):
                cpred = net.dose(ci, torch.zeros(ci.shape[0], dtype=torch.long, device=dev))
                L_ct = weighted_l1(cpred, cd, hi_frac, hi_w, grad_w=grad_w, het_w=het_w, lung_w=lung_w,
                                   density=cdens if (het_w or lung_w) else None,
                                   lung_lo=lung_lo, lung_hi=lung_hi)
            opt.zero_grad(set_to_none=True)
            scaler.scale(L_ct).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); _ema_update(ema, net)
            if step % int(cfg.get("log_every", 100)) == 0 or (a.smoke and step <= 3):
                print(f"step {step}/{max_steps} | CT-mix L {float(L_ct.detach()):.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
                if run: _wlog(run, {"ct_mix_l1": float(L_ct.detach()), "lr": sched.get_last_lr()[0]}, step)
            if a.smoke and step >= a.smoke:
                print(f"[smoke] {a.smoke} steps OK (ct-mix path)", flush=True); return
            ve = int(cfg.get("val_every", 0) or 0)
            if ve and step % ve == 0:
                vmae = validate(); tag = " *best*" if vmae < best else ""
                if vmae < best: best = vmae; _save(run_dir, net, opt, sched, scaler, ema, step, fname="best.pt")
                print(f"[val] step {step} | dose masked-MAE {vmae:.3f}%{tag}", flush=True)
                if run: _wlog(run, {"val_masked_mae": vmae, "val_best": best}, step)
            if step % int(cfg.get("ckpt_every", 2500)) == 0:
                _save(run_dir, net, opt, sched, scaler, ema, step)
            continue
        if _PROF: torch.cuda.synchronize(); _tp0 = time.time()
        pid = rng.choice(train_pts)
        if _PROF: _pre_hit = pid in vol_cache
        V = load_vol(V_pid := pid); rs = rays_of(pid)
        if _PROF: _t_load = time.time() - _tp0
        with torch.autocast("cuda"):
            sct01, densn = density_native(V, aug=True)   # MR-intensity aug active iff cfg.mr_aug set
            names = rng.sample(bfiles[pid], min(k_cps, len(bfiles[pid])))
            L_dose = 0.0; L_wepl = 0.0
            for name in names:
                z = np.load(cache / pid / name); ch = z["channels"].astype(np.float32)
                bb = tuple(int(v) for v in z["bbox"]); dose_np = z["dose"].astype(np.float32)
                z0, z1, y0, y1, x0, x1 = bb; dz, dy, dx = z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1
                fg = np.argwhere(dose_np > 0.1 * (dose_np.max() + 1e-12))
                s0, s1, s2, pz, py, px = _patch(dz, dy, dx, fg)
                pbb = (z0 + s0, z0 + s0 + pz - 1, y0 + s1, y0 + s1 + py - 1, x0 + s2, x0 + s2 + px - 1)
                csl = (slice(s0, s0 + pz), slice(s1, s1 + py), slice(s2, s2 + px))
                key = tuple(int(name[:-4].split("_")[i][1:]) for i in range(3))
                ray = rs[key]
                dens_c = densn[pbb[0]:pbb[1] + 1, pbb[2]:pbb[3] + 1, pbb[4]:pbb[5] + 1]
                wepl = _wepl_tensor(densn, V["spacing"], V["origin"], ray, pbb, pm.sad, dev)
                if wepl_w > 0 and wepl_dir:   # range-consistency: sCT WEPL -> real-CT (weplfix) WEPL
                    wfx = np.load(Path(wepl_dir) / pid / name)["wepl"].astype(np.float32)[csl[0], csl[1], csl[2]]
                    L_wepl = L_wepl + (wepl - torch.from_numpy(wfx).to(dev)).abs().mean() / 30.0
                pb = pb_fn(_Img(V), ray["ray_source"], ray["ray_target"], ray["e"], out_bbox=pbb,
                           machine=pm, density_override=densn, device=dev, return_tensor=True)
                cht = torch.from_numpy(ch[:, csl[0], csl[1], csl[2]]).to(dev)
                gt_dens = cht[0]                                            # GT-CT density (for loss weighting)
                inp = torch.stack([dens_c, wepl, pb * PROTON_DOSE_SCALE, cht[2], cht[3]], 0) / _PCS
                Z, Y, X = inp.shape[-3:]
                inp = F.pad(inp[None], (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
                pred = net.dose(inp, mod)[0, 0, :Z, :Y, :X]
                gt = torch.from_numpy(dose_np[csl]).to(dev) * PROTON_DOSE_SCALE
                L_dose = L_dose + weighted_l1(pred[None, None], gt[None, None], hi_frac, hi_w,
                                              grad_w=grad_w, het_w=het_w, lung_w=lung_w,
                                              density=gt_dens[None, None] if (het_w or lung_w) else None,
                                              lung_lo=lung_lo, lung_hi=lung_hi)
            L_dose = L_dose / len(names)
            anchor_gt = torch.from_numpy(V["dens01" if density_direct else "ct01"]).to(dev)
            L_sct = sct_anchor_loss(sct01, anchor_gt, w_bone=sct_wb, w_lung=sct_wl, thr=_dthr)
            L_wepl = L_wepl / len(names) if wepl_w > 0 else 0.0
            loss = L_dose + lam * L_sct + wepl_w * L_wepl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); _ema_update(ema, net)
        if _PROF:
            torch.cuda.synchronize(); _t_step = time.time() - _tp0
            print(f"[prof] step {step} total {_t_step*1000:.0f}ms | load_vol+rays {_t_load*1000:.0f}ms "
                  f"({'HIT' if _pre_hit else 'MISS'}) | compute {(_t_step-_t_load)*1000:.0f}ms", flush=True)
        if step % int(cfg.get("log_every", 100)) == 0 or (a.smoke and step <= 3):
            mem = torch.cuda.max_memory_allocated() / 1e9
            ld, ls = float(L_dose.detach()), float(L_sct.detach())
            lw = float(L_wepl.detach()) if (wepl_w > 0 and not isinstance(L_wepl, float)) else 0.0
            print(f"step {step}/{max_steps} | loss {loss.item():.4f} | dose {ld:.4f} | "
                  f"sct {ls:.4f} | wepl {lw:.4f} | lr {sched.get_last_lr()[0]:.2e} | peak {mem:.1f}G", flush=True)
            if run: _wlog(run, {"loss": loss.item(), "dose_l1": ld, "sct_l1": ls, "wepl_l1": lw,
                                "lr": sched.get_last_lr()[0]}, step)
        if a.smoke and step >= a.smoke:
            print(f"[smoke] {a.smoke} steps OK in {time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
            return
        ve = int(cfg.get("val_every", 0) or 0)
        if ve and step % ve == 0:
            vmae = validate()
            tag = " *best*" if vmae < best else ""
            if vmae < best:
                best = vmae; _save(run_dir, net, opt, sched, scaler, ema, step, fname="best.pt")
            print(f"[val] step {step} | dose masked-MAE {vmae:.3f}%{tag}", flush=True)
            if run: _wlog(run, {"val_masked_mae": vmae, "val_best": best}, step)
        if step % int(cfg.get("ckpt_every", 2500)) == 0:
            _save(run_dir, net, opt, sched, scaler, ema, step)
    _save(run_dir, net, opt, sched, scaler, ema, max_steps)
    print("done", flush=True)


if __name__ == "__main__":
    main()
