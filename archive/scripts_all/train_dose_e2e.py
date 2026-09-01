"""End-to-end dose-aware MRI->sCT->dose training.

MRI -> v4 MONAI UNet -> sCT(HU) -> diff_channels (density/rdepth/fluence/naive, differentiable)
-> v13 DoseUNet3D -> per-CP dose. Both nets warm-started; joint finetune with a multi-task loss
(dose weighted-L1 + lam_sct * sCT-anchor L1 vs real CT). The dose loss back-props THROUGH the
analytical physics into the synthesis net => the sCT becomes dose-aware.

Custom in-process loop (no DataLoader): per step pick one patient, run the FULL-volume synthesis once
(amortised over k_cps control points), then the differentiable ray-trace + dose net per CP.
"""
from __future__ import annotations
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import yaml
from monai.networks.nets import UNet

from doserad.model.unet3d import DoseUNet3D
from doserad.physics.diff_channels import (hu_to_density_torch, radiological_depth_fast_torch,
                                           fluence_torch, naive_dose_torch)
from doserad.physics.diff_channels_skinentry import radiological_depth_skinentry_torch
from doserad.physics.density import hu_to_density
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.machine import load_photon_machine
from doserad.beam.parse import load_photon_plan
from doserad.losses.dose_loss import weighted_l1
from doserad.data.dataset import _CH_SCALE, _NAIVE_SCALE, DOSE_SCALE, PhotonCropDataset
from doserad.io.mha import load_mha
from doserad.train.loop import build_optim, _ema_update, _save, _init_wandb, _wlog

CT_LO, CT_HI = -1000.0, 2000.0       # v4 sCT HU window -> [0,1]; denorm = x*3000-1000
DENS_MAX = 2.5                       # density-direct: synth out [0,1] -> density [0, DENS_MAX] g/cm^3


def _pad16(n): return (16 - n % 16) % 16


# HU-band thresholds expressed in CT-[0,1] space ((HU - CT_LO)/(CT_HI - CT_LO))
_T_BODY = (-500.0 - CT_LO) / (CT_HI - CT_LO)   # body  (HU > -500)
_T_LUNG = (-300.0 - CT_LO) / (CT_HI - CT_LO)   # lung/air (HU < -300, in body)
_T_BONE = (200.0 - CT_LO) / (CT_HI - CT_LO)    # bone  (HU > 200)


def sct_anchor_loss(sct01, ct01, w_bone=1.0, w_lung=1.0, thr=None):
    """L1 sCT anchor, optionally upweighting bone (CT>200 HU) and lung/air (CT<-300 HU, in body).
    Lever 3 (region-weighted): bone bias and lung variance are the dominant sCT errors. With
    w_bone=w_lung=1.0 this reduces to plain mean L1. `thr=(body,lung,bone)` overrides the HU-01
    thresholds (used for density-direct where sct01/ct01 live in density-01 space)."""
    err = (sct01 - ct01).abs()
    if w_bone == 1.0 and w_lung == 1.0:
        return err.mean()
    tb, tl, tn = thr if thr is not None else (_T_BODY, _T_LUNG, _T_BONE)
    body = ct01 > tb
    bone = ct01 > tn
    lung = body & (ct01 < tl)
    w = torch.ones_like(ct01) + (w_bone - 1.0) * bone.float() + (w_lung - 1.0) * lung.float()
    return (err * w).sum() / w.sum()


class E2E(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.synth = UNet(spatial_dims=3, in_channels=int(cfg.get("synth_in_ch", 1)), out_channels=1,
                          channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
                          num_res_units=2, norm="INSTANCE", act="LEAKYRELU")
        self.dose = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                               bottleneck=cfg.get("bottleneck", "dilated"))

    def sct01(self, mri01, ckpt=True):              # mri01 (1,1,Z,Y,X) -> sCT in [0,1] full grid
        Z, Y, X = mri01.shape[-3:]
        xp = F.pad(mri01, (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
        # gradient-checkpoint the full-volume synthesis (its activations are the peak-memory driver;
        # recomputed in backward -> large peak-memory saving for the joint graph).
        if ckpt and self.training:
            out = torch.utils.checkpoint.checkpoint(self.synth, xp, use_reentrant=False)
        else:
            out = self.synth(xp)
        return out[..., :Z, :Y, :X]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0, help="run N smoke steps (k_cps=1) and exit")
    ap.add_argument("--resume", action="store_true",
                    help="resume model/opt/sched/step from <run_dir>/state.pt (for quiet-hour kills)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda"
    # skin-entry consistency with the CT path: entered-gated depth + entered-gated naive prior.
    # rdepth_fn picks the skin-entry torch depth (rdepth==0 upstream of skin) when enabled; the
    # naive prior is then gated by (rdepth>0) via naive_dose_torch(skin_gate=...). Identical
    # semantics to numpy compute_naive_dose(skin_gate) / raytrace_skinentry (skin_thr=0.05).
    skin_entry = bool(cfg.get("naive_skin_gate", False))
    rdepth_fn = radiological_depth_skinentry_torch if skin_entry else radiological_depth_fast_torch
    ROOT = cfg["root_dir"]
    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    anchors = machine.hu_anchors
    run_dir = Path(cfg["run_root"]) / cfg["exp_name"]; run_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["cache_dir"])
    max_steps = a.max_steps or cfg["max_steps"]
    k_cps = 1 if a.smoke else int(cfg.get("k_cps", 2))
    lam = float(cfg.get("lam_sct", 0.5))
    lung_dens_aug = float(cfg.get("lung_dens_aug", 0.0))   # B: lung density-robustness augmentation
    sct_wb = float(cfg.get("sct_w_bone", 1.0))      # lever 3: upweight bone in sCT anchor (1.0 = off)
    sct_wl = float(cfg.get("sct_w_lung", 1.0))      # lever 3: upweight lung/air in sCT anchor
    coarse_dir = cfg.get("coarse_dir", None)        # classify-then-regress: 2-ch synth [MR, coarse CT]
    density_direct = bool(cfg.get("density_direct", False))   # synth outputs density (skip HU bottleneck)
    _dthr = tuple(float(hu_to_density(np.array([h], np.float32), anchors)[0] / DENS_MAX)
                  for h in (-500.0, -300.0, 200.0)) if density_direct else None   # density-01 anchor thresholds
    P = int(cfg.get("patch", 128))
    freeze = bool(cfg.get("freeze_synth", False))   # control: fix sCT at v4, only finetune the dose net
    mr_aug_cfg = cfg.get("mr_aug", None)   # MR-intensity aug (bias/gamma/noise) on the TRAIN synth input
    # ONLY (never GT dose/geometry, never validation) — instils cross-scanner intensity invariance in the
    # E2E synth+dose, matching the shift-robust clf. Safe per augmentation-rules (intensity, not geometric).

    def _aug_mri(mri):
        if not mr_aug_cfg:
            return mri
        Z, Y, X = mri.shape[-3:]
        amp = float(np.random.uniform(0.0, float(mr_aug_cfg.get("bias", 0.5))))
        small = torch.randn(1, 1, max(Z // 8, 2), max(Y // 8, 2), max(X // 8, 2), device=dev)
        b = F.interpolate(small, size=(Z, Y, X), mode="trilinear", align_corners=False)
        b = 1.0 + amp * (b / (b.abs().amax() + 1e-6))
        mri = (mri * b)
        mri = (mri / (mri.amax() + 1e-6)).clamp(0, 1)                       # renorm (survives pct-norm)
        gl, gh = mr_aug_cfg.get("gamma", [0.7, 1.6])
        mri = mri.clamp(0, None) ** float(np.random.uniform(gl, gh))        # contrast
        sn = float(mr_aug_cfg.get("noise", 0.03))
        if sn > 0:
            mri = (mri + float(np.random.uniform(0, sn)) * torch.randn_like(mri)).clamp(0, 1)
        return mri

    splits = json.load(open(cfg["splits"])); fold = splits[f"fold_{cfg['fold']}"]
    train_pts = [p for p in fold["train"] if (cache / p).is_dir()]
    val_pts = [p for p in fold["val"] if (cache / p).is_dir()]
    # per-patient CP file list + plan + geometry cache
    cpfiles = {p: sorted(f.name for f in (cache / p).glob("*.npz")) for p in train_pts + val_pts}
    # fixed validation set of CPs (masked-MAE % proxy; the real metric is plan-γ via eval_dose_e2e)
    _vr = random.Random(123)
    val_list = [(p := _vr.choice(val_pts), _vr.choice(cpfiles[p])) for _ in range(int(cfg.get("val_cps", 32)))]
    plans = {}
    def plan_of(pid):
        if pid not in plans:
            pl = load_photon_plan(Path(ROOT) / pid / f"{pid}.json")
            geo = {}
            for b in pl.beams:
                for cp in b.control_points:
                    iso = np.asarray(b.iso_center, np.float64)
                    src = beam_source_pos(iso, machine.sad_mm, cp.gantry_angle)
                    ax, u, v = beam_basis(cp.gantry_angle)
                    geo[f"{b.beam_idx}_{cp.cp_idx:03d}"] = (iso, src, ax, u, v)
            plans[pid] = geo
        return plans[pid]

    vol_cache = {}
    def load_vol(pid):
        if pid not in vol_cache:
            mr = load_mha(Path(ROOT) / pid / "image" / "mr.mha")
            a_mr = mr.array.astype(np.float32)
            lo, hi = np.percentile(a_mr, 1), np.percentile(a_mr, 99)
            mr01 = np.clip((a_mr - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)
            ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
            ct01 = np.clip((ct.array.astype(np.float32) - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)
            dens = hu_to_density(ct.array, anchors).astype(np.float32)   # GT-CT density for loss weighting
            # full-grid world coords for the ray-trace (constant per patient)
            sx, sy, sz = mr.spacing; ox, oy, oz = mr.origin
            nz, ny, nx = a_mr.shape
            xs = ox + np.arange(nx) * sx; ys = oy + np.arange(ny) * sy; zs = oz + np.arange(nz) * sz
            gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
            coords = np.stack([gx, gy, gz], -1).astype(np.float32)
            entry = dict(mr01=mr01, ct01=ct01, dens=dens, dens01=np.clip(dens / DENS_MAX, 0, 1).astype(np.float32),
                         spacing=mr.spacing, origin=mr.origin)
            if coarse_dir:                                  # classify-then-regress: 2nd synth input
                cv = load_mha(Path(coarse_dir) / f"{pid}.nii.gz").array.astype(np.float32)
                entry["coarse01"] = np.clip((cv - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)
            vol_cache[pid] = entry
            if len(vol_cache) > 8:                         # bounded LRU
                vol_cache.pop(next(iter(vol_cache)))
        return vol_cache[pid]

    net = E2E(cfg).to(dev)
    # warm-start
    sd = torch.load(cfg["synth_ckpt"], map_location=dev)
    net.synth.load_state_dict(sd["net"]); print(f"[warm] synth <- v4 (epoch {sd.get('epoch')})", flush=True)
    dd = torch.load(cfg["dose_ckpt"], map_location=dev)
    dstate = dd.get("ema", dd.get("model"))
    img_ch = int(cfg["in_ch"]) > 6                  # B: append raw MRI + sCT as dose-net input channels
    if img_ch:
        sw = dstate["stem.weight"]                  # v13 [base,6,3,3,3]
        new = net.dose.state_dict()["stem.weight"].clone().zero_()   # [base,in_ch,3,3,3]
        new[:, :sw.shape[1]] = sw                   # first 6 = v13; extra (MRI,sCT) = 0
        dstate = dict(dstate); dstate["stem.weight"] = new
        print(f"[warm] dose stem 6->{cfg['in_ch']}ch (extra MRI+sCT channels zero-init)", flush=True)
    net.dose.load_state_dict(dstate); print("[warm] dose <- v13ft EMA", flush=True)
    if cfg.get("init_from"):   # warm-start the FULL E2E (synth+dose) from a prior fusion checkpoint
        ist = torch.load(cfg["init_from"], map_location=dev)
        net.load_state_dict(ist.get("ema", ist.get("model")))
        print(f"[init] full E2E <- {cfg['init_from']}", flush=True)
    if freeze:
        for p in net.synth.parameters():
            p.requires_grad_(False)
        net.synth.eval(); print("[freeze] synth FROZEN (sCT fixed at v4; only dose net finetunes)", flush=True)
    if bool(cfg.get("freeze_dose", False)):          # C: fix v13 dose engine; only synth trains via dose loss
        for p in net.dose.parameters():
            p.requires_grad_(False)
        print("[freeze] dose net FROZEN (v13 fixed critic; only synth trains, dose grad flows through it)", flush=True)

    opt, sched, scaler, ema = build_optim(net, max_steps, lr=cfg["lr"], wd=cfg["weight_decay"])
    run = _init_wandb(cfg, run_dir) if cfg.get("wandb") and not a.smoke else None
    rng = random.Random(0)
    mod = torch.zeros(1, dtype=torch.long, device=dev)

    lwk = dict(hi_frac=cfg["hi_frac"], hi_w=cfg["hi_w"], grad_w=cfg["grad_w"],
               het_w=cfg["het_w"], lung_w=cfg["lung_w"], lung_lo=cfg["lung_lo"], lung_hi=cfg["lung_hi"])

    # MULTI-MODAL (CT-mix, opt-in via cfg.ct_mix_cache): interleave real-CT cached-channel batches that
    # update ONLY the dose net (synth not involved in a CT step), so the dose net trains on BOTH real-CT
    # and synthesized-CT densities. Off by default -> byte-identical to the standard MRI e2e training.
    ct_iter = None; ct_mix_prob = float(cfg.get("ct_mix_prob", 0.5))
    if cfg.get("ct_mix_cache"):
        from torch.utils.data import DataLoader
        ctc = Path(cfg["ct_mix_cache"]); ct_rows = []
        for p in train_pts:
            for f in sorted((ctc / p).glob("*.npz")):
                bi, ci = f.stem.split("_"); ct_rows.append({"patient_id": p, "beam_idx": int(bi), "cp_idx": int(ci)})
        ct_ds = PhotonCropDataset(ct_rows, str(ctc), patch=(P, P, P), fg_prob=float(cfg.get("fg_prob", 0.7)),
                                  add_naive=True, naive_skin_gate=cfg.get("naive_skin_gate"))
        ct_loader = DataLoader(ct_ds, batch_size=int(cfg.get("ct_mix_batch", 1)), shuffle=True,
                               num_workers=0, drop_last=True)
        def _ct_gen():
            while True:
                for bb in ct_loader: yield bb
        ct_iter = _ct_gen()
        print(f"[ct-mix] {len(ct_rows)} CT crops from {ctc.name}, prob {ct_mix_prob} (dose-net only)", flush=True)

    from collections import defaultdict
    from monai.metrics import SSIMMetric
    _ssim = SSIMMetric(spatial_dims=3, data_range=1.0)
    @torch.no_grad()
    def validate():                                  # dose masked-MAE % + sCT HU-MAE/SSIM vs real CT
        net.eval()
        byp = defaultdict(list)
        for p, c in val_list:
            byp[p].append(c)
        tot, n = 0.0, 0
        hu_s, ss_s, npat = 0.0, 0.0, 0
        bone_s, lung_s = 0.0, 0.0
        for pid, cps in byp.items():
            Vv = load_vol(pid); g = plan_of(pid)
            mriv = torch.from_numpy(Vv["mr01"]).to(dev)[None, None]
            synv = mriv if not coarse_dir else torch.cat(
                [mriv, torch.from_numpy(Vv["coarse01"]).to(dev)[None, None]], 1)
            with torch.autocast("cuda"):
                sctv = net.sct01(synv, ckpt=False)[0, 0]
                # sCT image quality vs real CT (per unique val patient)
                ct01 = torch.from_numpy(Vv["ct01"]).to(dev)
                body = ct01 > _T_BODY                # in-body (HU > -500 in the [-1000,2000]->[0,1] window)
                err = (sctv - ct01).abs()
                hu_s += float(err[body].mean()) * 3000.0
                bonem = ct01 > _T_BONE; lungm = body & (ct01 < _T_LUNG)
                bone_s += float(err[bonem].mean()) * 3000.0 if bonem.any() else 0.0
                lung_s += float(err[lungm].mean()) * 3000.0 if lungm.any() else 0.0
                ss_s += float(_ssim(sctv[None, None].float(), ct01[None, None].float()))
                npat += 1
                densv = sctv.clamp(0, 1) * DENS_MAX if density_direct else hu_to_density_torch(sctv * (CT_HI - CT_LO) + CT_LO, anchors)
                for name in cps:
                    z = np.load(cache / pid / name); ch = z["channels"].astype(np.float32)
                    bb = tuple(int(v) for v in z["bbox"]); z0, z1, y0, y1, x0, x1 = bb
                    sl = (slice(z0, z1+1), slice(y0, y1+1), slice(x0, x1+1))
                    iso, src, ax, u, v = g[name[:-4]]
                    rd = rdepth_fn(densv, Vv["spacing"], Vv["origin"], src, ax, u, v, iso, out_bbox=bb)
                    cht = torch.from_numpy(ch).to(dev); fl = fluence_torch((cht[2] > 0).float(), rd)
                    nv = naive_dose_torch(densv[sl], rd, fl, cht[4], skin_gate=skin_entry)
                    cl = [densv[sl] / _CH_SCALE[0], rd / _CH_SCALE[1], fl / _CH_SCALE[2],
                          cht[3] / _CH_SCALE[3], cht[4] / _CH_SCALE[4], nv / _NAIVE_SCALE]
                    if img_ch:
                        cl += [mriv[0, 0][sl], sctv[sl]]
                    ip = torch.stack(cl, 0); Z, Y, X = ip.shape[-3:]
                    ip = F.pad(ip[None], (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
                    pred = net.dose(ip, mod)[0, 0, :Z, :Y, :X].float() / DOSE_SCALE
                    gt = torch.from_numpy(z["dose"].astype(np.float32)).to(dev)
                    m = gt >= 0.1 * gt.max()
                    if m.any():
                        tot += float((pred[m] - gt[m]).abs().mean() / gt[m].mean().clamp_min(1e-9)) * 100; n += 1
        net.train()
        if freeze:
            net.synth.eval()
        return (tot / max(n, 1), hu_s / max(npat, 1), ss_s / max(npat, 1),
                bone_s / max(npat, 1), lung_s / max(npat, 1))

    start_step = 0
    if a.resume and (run_dir / "state.pt").exists():
        rst = torch.load(run_dir / "state.pt", map_location=dev)
        net.load_state_dict(rst["model"]); ema.load_state_dict(rst["ema"])
        opt.load_state_dict(rst["opt"]); sched.load_state_dict(rst["sched"])
        scaler.load_state_dict(rst["scaler"]); start_step = int(rst["step"])
        print(f"[resume] from step {start_step} ({run_dir}/state.pt)", flush=True)

    best = float("inf")
    t0 = time.time()
    for step in range(start_step + 1, max_steps + 1):
        if ct_iter is not None and rng.random() < ct_mix_prob:   # multi-modal real-CT step (dose net only)
            bb = next(ct_iter); ci = bb["input"].to(dev); cd = bb["dose"].to(dev)
            with torch.autocast("cuda"):
                cpred = net.dose(ci, torch.zeros(ci.shape[0], dtype=torch.long, device=dev))
                L_ct = weighted_l1(cpred, cd, density=ci[:, :1] * _CH_SCALE[0], **lwk)
            opt.zero_grad(set_to_none=True); scaler.scale(L_ct).backward()
            scaler.step(opt); scaler.update(); sched.step(); _ema_update(ema, net)
            if step % int(cfg.get("log_every", 100)) == 0:
                print(f"step {step}/{max_steps} | CT-mix L {L_ct.item():.4f} | lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if a.smoke and step >= a.smoke:
                print(f"[smoke] {a.smoke} steps OK (ct-mix path)", flush=True); return
            ve = int(cfg.get("val_every", 0) or 0)
            if ve and step % ve == 0:
                vmae, vhu, vss, vbone, vlung = validate(); tag = " *best*" if vmae < best else ""
                if vmae < best: best = vmae; _save(run_dir, net, opt, sched, scaler, ema, step, fname="best.pt")
                print(f"[val] step {step} | dose masked-MAE {vmae:.3f}% | sCT HU-MAE {vhu:.1f}{tag}", flush=True)
            if step % int(cfg.get("ckpt_every", 2500)) == 0: _save(run_dir, net, opt, sched, scaler, ema, step)
            continue
        pid = rng.choice(train_pts)
        V = load_vol(pid); geo = plan_of(pid)
        mri = torch.from_numpy(V["mr01"]).to(dev)[None, None]
        mri = _aug_mri(mri)                          # MR-intensity aug (train only; no-op if mr_aug unset)
        syn_in = mri if not coarse_dir else torch.cat(
            [mri, torch.from_numpy(V["coarse01"]).to(dev)[None, None]], 1)   # (1,2,Z,Y,X)
        with torch.autocast("cuda"):
            if freeze:
                with torch.no_grad():
                    sct01 = net.sct01(syn_in, ckpt=False)[0, 0]   # fixed sCT (no synth grad)
            else:
                sct01 = net.sct01(syn_in)[0, 0]                   # (Z,Y,X) [0,1], grad through synth
            if density_direct:
                density = sct01.clamp(0, 1) * DENS_MAX          # synth out -> density (clamped; UNet is unbounded)
            else:
                density = hu_to_density_torch(sct01 * (CT_HI - CT_LO) + CT_LO, anchors)  # (Z,Y,X) grad
            if lung_dens_aug > 0:   # B: random ±aug scale on lung-region density (GT-CT lung mask),
                                    # GT dose unchanged -> dose net learns robustness to lung density error
                gd = torch.from_numpy(V["dens"]).to(dev)
                lmask = (gd > 0.02) & (gd < 0.6)
                f = 1.0 + (torch.rand((), device=dev) * 2 - 1) * lung_dens_aug
                density = density * torch.where(lmask, f, torch.ones((), device=dev))
            cps = rng.sample(cpfiles[pid], min(k_cps, len(cpfiles[pid])))
            L_dose = 0.0
            for name in cps:
                z = np.load(cache / pid / name)
                ch = z["channels"].astype(np.float32); bb = tuple(int(v) for v in z["bbox"])
                dose_np = z["dose"].astype(np.float32)
                z0, z1, y0, y1, x0, x1 = bb
                dz, dy, dx = z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1
                # sample a foreground-biased patch within the crop (matches v13 128^3 patch training;
                # bounds dose-net + rdepth memory so the largest open-field bboxes don't OOM)
                pz, py, px = min(P, dz), min(P, dy), min(P, dx)
                fg = np.argwhere(dose_np > 0.1 * (dose_np.max() + 1e-12))
                if len(fg):
                    c = fg[rng.randrange(len(fg))]
                    sz0 = int(np.clip(c[0] - pz // 2, 0, dz - pz))
                    sy0 = int(np.clip(c[1] - py // 2, 0, dy - py))
                    sx0 = int(np.clip(c[2] - px // 2, 0, dx - px))
                else:
                    sz0, sy0, sx0 = (dz - pz) // 2, (dy - py) // 2, (dx - px) // 2
                # patch bbox in FULL-volume coords (for the differentiable ray-trace output)
                pbb = (z0 + sz0, z0 + sz0 + pz - 1, y0 + sy0, y0 + sy0 + py - 1, x0 + sx0, x0 + sx0 + px - 1)
                fsl = (slice(pbb[0], pbb[1] + 1), slice(pbb[2], pbb[3] + 1), slice(pbb[4], pbb[5] + 1))
                csl = (slice(sz0, sz0 + pz), slice(sy0, sy0 + py), slice(sx0, sx0 + px))   # within crop
                iso, src, ax, u, v = geo[name[:-4]]
                rdepth = rdepth_fn(density, V["spacing"], V["origin"], src, ax, u, v,
                                   iso, out_bbox=pbb)
                dens_c = density[fsl]
                cht = torch.from_numpy(ch[:, csl[0], csl[1], csl[2]]).to(dev)
                dist_c = cht[3]; source_c = cht[4]; open_c = (cht[2] > 0).float()
                fl = fluence_torch(open_c, rdepth)
                naive = naive_dose_torch(dens_c, rdepth, fl, source_c, skin_gate=skin_entry)
                chans = [dens_c / _CH_SCALE[0], rdepth / _CH_SCALE[1], fl / _CH_SCALE[2],
                         dist_c / _CH_SCALE[3], source_c / _CH_SCALE[4], naive / _NAIVE_SCALE]
                if img_ch:                                   # B: + raw MRI + predicted sCT (both [0,1])
                    chans += [mri[0, 0][fsl], sct01[fsl]]
                inp = torch.stack(chans, 0)
                Z, Y, X = inp.shape[-3:]
                inp = F.pad(inp[None], (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
                pred = net.dose(inp, mod)[0, 0, :Z, :Y, :X]
                dose = torch.from_numpy(dose_np[csl]).to(dev)
                densw = torch.from_numpy(V["dens"][fsl]).to(dev)
                L_dose = L_dose + weighted_l1(pred[None, None], (dose * DOSE_SCALE)[None, None],
                                              density=densw[None, None], **lwk)
            L_dose = L_dose / len(cps)
            anchor_gt = torch.from_numpy(V["dens01" if density_direct else "ct01"]).to(dev)
            L_sct = sct_anchor_loss(sct01, anchor_gt, w_bone=sct_wb, w_lung=sct_wl, thr=_dthr)
            loss = L_dose if freeze else (L_dose + lam * L_sct)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step(); _ema_update(ema, net)
        if step % int(cfg.get("log_every", 100)) == 0 or (a.smoke and step <= 3):
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step}/{max_steps} | loss {loss.item():.4f} | dose {float(L_dose):.4f} | "
                  f"sct {float(L_sct):.4f} | lr {sched.get_last_lr()[0]:.2e} | peak {mem:.1f}G", flush=True)
            if run: _wlog(run, {"loss": loss.item(), "dose_l1": float(L_dose), "sct_l1": float(L_sct),
                                "lr": sched.get_last_lr()[0]}, step)
        if a.smoke and step >= a.smoke:
            print(f"[smoke] {a.smoke} steps OK in {time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
            return
        ve = int(cfg.get("val_every", 0) or 0)
        if ve and step % ve == 0:
            vmae, vhu, vss, vbone, vlung = validate()
            tag = " *best*" if vmae < best else ""
            if vmae < best:
                best = vmae; _save(run_dir, net, opt, sched, scaler, ema, step, fname="best.pt")
            print(f"[val] step {step} | dose masked-MAE {vmae:.3f}% | sCT HU-MAE {vhu:.1f} "
                  f"(bone {vbone:.0f} / lung {vlung:.0f}) | sCT SSIM {vss:.3f}{tag}", flush=True)
            if run: _wlog(run, {"val_masked_mae": vmae, "val_best": best, "val_sct_humae": vhu,
                                "val_sct_bone_mae": vbone, "val_sct_lung_mae": vlung, "val_sct_ssim": vss}, step)
        if step % int(cfg.get("ckpt_every", 2500)) == 0:
            _save(run_dir, net, opt, sched, scaler, ema, step)
    _save(run_dir, net, opt, sched, scaler, ema, max_steps)
    print("done", flush=True)


if __name__ == "__main__":
    main()
