"""Task-4 (Proton-MRI) PLAN-gamma baseline: MRI -> wholesoft sCT -> proton dose.

Zero-extra-training baseline. Pipeline per patient (16 held-out val, fold_0):
  mr.mha -> [mr01, coarse_whole_soft] -> wholesoft E2E synth -> sCT (HU) -> density
  per beamlet, recompute the DENSITY-dependent channels ON THE sCT density:
    ch0 = sCT density,  ch1 = WEPL (ray-march on sCT),  pb = masked GPU-PB prior on sCT
  reuse the GEOMETRY-only cached channels: lateral_dist, energy
  -> masked proton engine (DoseUNet3D) -> beamlet dose -> accumulate plan
  -> plan-level local gamma 1/1, 2/2, 3/3 vs the MC GT proton dose.

MR/CT are co-registered (same grid) so the sCT density drops straight onto the proton
cache/plan grid. The GT proton dose is the same MC GT as Task 3 (computed on real CT);
here we PREDICT it from MRI-derived sCT (Task-2-of-Task-1 analogue).

Usage:
  python scripts/eval_protonmri_plan.py \
    --synth-config configs/experiments/mri_dose_e2e_fuse_a14_wholesoft.yaml \
    --synth-ckpt   <runs>/mri_dose_e2e_fuse_a14_wholesoft/best.pt \
    --engine-config configs/experiments/proton_ct_prior_gpu_masked_scratch.yaml \
    --engine-ckpt   <runs>/proton_ct_prior_gpu_masked_scratch/best.pt \
    --label protonmri_wholesoft_masked
"""
from __future__ import annotations
import argparse, json, sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for train_dose_e2e import
import statistics as st
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.gamma import gamma_array, gamma_pass
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import (ProtonMachineData, proton_pb_dose_gpu,
                                           _compute_ssd, _wepl_crop)
from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry
from train_dose_e2e import E2E, CT_LO, CT_HI

# --- DIAGNOSTIC hooks (env-gated; default OFF -> unchanged behaviour) ---
# DOSERAD_MR_SHIFT=1 : apply a realistic cross-institution MR intensity/contrast shift (bias field +
#   post-norm gamma, both survive the pct-1/99 norm) before the sCT pipeline.
# DOSERAD_SHIFT_CLF=<clf.pt> : compute the coarse prior LIVE from the (possibly shifted) MR via this
#   classifier — matches the container (which runs clf live) and captures the clf's degradation under
#   shift, unlike the default precomputed (clean, in-sample) coarse.
_MR_SHIFT = os.environ.get("DOSERAD_MR_SHIFT")
_SHIFT_CLF_PATH = os.environ.get("DOSERAD_SHIFT_CLF")
_shift_state = {"clf": None}


def _apply_mr_shift(a_mr):
    from scipy.ndimage import gaussian_filter, zoom
    amp = float(os.environ.get("DOSERAD_SHIFT_BIAS", "0.25"))
    gam = float(os.environ.get("DOSERAD_SHIFT_GAMMA", "1.25"))
    rng = np.random.default_rng(0)
    small = rng.normal(0, 1, [max(s // 8, 2) for s in a_mr.shape]).astype(np.float32)
    small = gaussian_filter(small, 2.0)
    f = zoom(small, [a_mr.shape[i] / small.shape[i] for i in range(3)], order=1)
    bias = 1.0 + amp * (f / (np.abs(f).max() + 1e-6))
    ap = a_mr * bias
    lo, hi = np.percentile(ap, 1), np.percentile(ap, 99)
    n = np.clip((ap - lo) / max(hi - lo, 1.0), 0, 1)
    return (np.clip(n, 0, None) ** gam) * (hi if hi > 0 else 1.0)   # post-norm gamma, re-scaled


def _grid_roundtrip(mr_sitk):
    """DIAGNOSTIC: the REAL proton-MRI test MR arrives at NATIVE 1x1x3 mm (proton grid) and the
    container resamples it to 2 mm for the synth; our internal eval reads native-2 mm MR and never
    exercises that resample. Approximate the container's grid handling on our 2 mm MR: 2 mm -> 1x1x3
    -> back to the ORIGINAL 2 mm grid (preserves size/origin so CopyInformation stays valid)."""
    def grid(ref, sp):
        size = [int(round(ref.GetSize()[i] * ref.GetSpacing()[i] / sp[i])) for i in range(3)]
        g = sitk.Image(size, sitk.sitkFloat32)
        g.SetOrigin(ref.GetOrigin()); g.SetDirection(ref.GetDirection()); g.SetSpacing(sp)
        return g
    m13 = sitk.Resample(mr_sitk, grid(mr_sitk, (1., 1., 3.)), sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    m2 = sitk.Resample(m13, mr_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(m2).astype(np.float32)


def _live_coarse(a_mr, dev):
    from container.mri_synth import load_classifier, _norm_mr01, REP_HU, _pad16
    if _shift_state["clf"] is None:
        _shift_state["clf"] = load_classifier(_SHIFT_CLF_PATH, dev)
    clf = _shift_state["clf"]
    mrx = torch.from_numpy(np.transpose(_norm_mr01(a_mr), (2, 1, 0)))[None, None].to(dev)
    X, Y, Z = mrx.shape[-3:]
    xp = F.pad(mrx, (0, _pad16(Z), 0, _pad16(Y), 0, _pad16(X)))
    p = torch.softmax(clf(xp)[..., :X, :Y, :Z].float(), 1)[0]
    coarse = (p * torch.from_numpy(REP_HU).to(dev).view(-1, 1, 1, 1)).sum(0).permute(2, 1, 0)
    return torch.clamp((coarse - CT_LO) / (CT_HI - CT_LO), 0, 1)


ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"   # 2mm MR grid the sCT synth was trained on
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"


def _wepl_on_density(image, density, ray_source, ray_target, bbox, pm, dev):
    """Corrected ray-march WEPL (machine-SAD source) computed on the GIVEN density (here sCT).
    Mirrors recompute_proton_wepl._correct_wepl."""
    sx, sy, sz = image.spacing; ox, oy, oz = image.origin
    tgt = np.asarray(ray_target, np.float64); jsrc = np.asarray(ray_source, np.float64)
    axis = tgt - jsrc; axis = axis / (np.linalg.norm(axis) + 1e-12)
    src = (tgt - axis * pm.sad).astype(np.float32)
    z0, z1, y0, y1, x0, x1 = bbox
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)
    ssd = _compute_ssd(density, image.spacing, image.origin, src, axis.astype(np.float32), pm.sad, dev)
    wepl = _wepl_crop(density, image.spacing, image.origin, src, coords, dev,
                      march_start_mm=max(ssd - 50.0, 0.0))
    if isinstance(wepl, torch.Tensor):
        wepl = wepl.detach().cpu().numpy()
    return np.asarray(wepl, np.float32)


@torch.no_grad()
def _predict(net, inp, dev):
    x = torch.from_numpy(inp[None]).to(dev)
    _, _, Z, Y, X = x.shape
    pz, py, px = (-Z) % 16, (-Y) % 16, (-X) % 16
    if pz or py or px:
        x = F.pad(x, (0, px, 0, py, 0, pz))
    with torch.autocast("cuda", enabled=(dev != "cpu")):
        y = net(x, torch.zeros(1, dtype=torch.long, device=dev))
    return (y[0, 0, :Z, :Y, :X].float() / PROTON_DOSE_SCALE).cpu().numpy()


@torch.no_grad()
def _make_sct_density(synth, scfg, pid, ct_sitk, hu_anchors, dev):
    """photon 2mm mr.mha + coarse_whole_soft -> wholesoft sCT (HU, 2mm) -> resample to the proton
    native grid (ct_sitk reference, same physical space) -> density. The synth was trained at 2mm,
    so sCT is generated at 2mm then resampled to the 1x1x3mm proton/plan grid."""
    mr_sitk = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/mr.mha")     # 2mm
    if os.environ.get("DOSERAD_MR_ROUNDTRIP"):       # DIAGNOSTIC: mimic native-1x1x3 -> 2mm resample
        a_mr = _grid_roundtrip(mr_sitk)
    else:
        a_mr = sitk.GetArrayFromImage(mr_sitk).astype(np.float32)
    if _MR_SHIFT:                                    # DIAGNOSTIC: simulate cross-institution MR shift
        a_mr = _apply_mr_shift(a_mr)
    lo, hi = np.percentile(a_mr, 1), np.percentile(a_mr, 99)
    mr01 = torch.from_numpy(np.clip((a_mr - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)).to(dev)
    if _SHIFT_CLF_PATH:                              # DIAGNOSTIC: live clf coarse (matches container)
        co = _live_coarse(a_mr, dev)
    else:
        cv = load_mha(Path(scfg["coarse_dir"]) / f"{pid}.nii.gz").array.astype(np.float32)
        co = torch.from_numpy(np.clip((cv - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)).to(dev)
    with torch.autocast("cuda", enabled=(dev != "cpu")):
        sct01 = synth.sct01(torch.stack([mr01, co], 0)[None])[0, 0]
    if scfg.get("density_direct"):       # synth out -> density directly; resample DENSITY to native
        DENS_MAX = 2.5
        dens_2mm = (sct01.float().clamp(0, 1) * DENS_MAX).cpu().numpy().astype(np.float32)
        img = sitk.GetImageFromArray(dens_2mm); img.CopyInformation(mr_sitk)
        res = sitk.Resample(img, ct_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        density = sitk.GetArrayFromImage(res).astype(np.float32)
        return density, density
    sct_hu_2mm = (sct01.float() * (CT_HI - CT_LO) + CT_LO).cpu().numpy().astype(np.float32)
    sct_img = sitk.GetImageFromArray(sct_hu_2mm); sct_img.CopyInformation(mr_sitk)
    res = sitk.Resample(sct_img, ct_sitk, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    sct_hu = sitk.GetArrayFromImage(res).astype(np.float32)           # native proton grid
    density = hu_to_density(sct_hu, hu_anchors).astype(np.float32)
    return density, sct_hu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-config", required=True)
    ap.add_argument("--synth-ckpt", default=None)
    ap.add_argument("--engine-config", required=True)
    ap.add_argument("--engine-ckpt", default=None)
    ap.add_argument("--e2e-ckpt", default=None,
                    help="single E2E checkpoint (dose-aware joint model) for BOTH synth and dose; "
                         "--synth-config/--engine-config should both be the proton e2e config")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--save-pred", default=None)
    ap.add_argument("--real-ct", action="store_true",
                    help="DIAGNOSTIC: use real-CT density (no synth/resample) — should reproduce Task-3 ~97; "
                         "isolates pipeline bugs from the sCT cost")
    ap.add_argument("--unmasked", action="store_true",
                    help="use the UNMASKED PB prior (no body mask) — for the unmasked champion engine "
                         "(masked prior was -0.9 vs unmasked on Proton-CT, so Task-4 uses unmasked)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    scfg = yaml.safe_load(open(a.synth_config))
    ecfg = yaml.safe_load(open(a.engine_config))
    skin_entry = bool(ecfg.get("skin_entry", False))   # per-ray entered gate (== CT proton); no post-hoc mask_air
    pb_fn = proton_pb_dose_gpu_skinentry if skin_entry else proton_pb_dose_gpu
    machine = load_photon_machine(MACHINE); hu_anchors = machine.hu_anchors
    pm = ProtonMachineData(device=dev)

    if a.e2e_ckpt:                       # dose-aware joint model: one E2E ckpt -> synth + dose
        e2e = E2E(scfg).to(dev).eval()
        sd = torch.load(a.e2e_ckpt, map_location=dev); e2e.load_state_dict(sd.get("ema", sd.get("model")))
        synth = e2e; eng = e2e.dose
    else:                                # baseline: separate frozen synth + frozen engine
        synth = None
        if not a.real_ct:                # --real-ct skips the synth entirely (real CT density)
            synth = E2E(scfg).to(dev).eval()
            ss = torch.load(a.synth_ckpt, map_location=dev); synth.load_state_dict(ss.get("ema", ss.get("model")))
        eng = DoseUNet3D(in_ch=ecfg["in_ch"], base=ecfg["base_ch"], levels=ecfg["levels"],
                         bottleneck=ecfg.get("bottleneck", "plain")).to(dev).eval()
        es = torch.load(a.engine_ckpt, map_location=dev); eng.load_state_dict(es.get("ema", es.get("model")))

    cache = Path(ecfg["cache_dir"])
    val = json.load(open(ecfg["splits"]))[f"fold_{ecfg['fold']}"]["val"]
    if a.only:
        want = set(a.only.split(",")); val = [p for p in val if p in want]
    if a.max_patients:
        val = val[:a.max_patients]

    rows = []
    for pid in val:
        t0 = time.time()
        ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")   # geometry only (grid == MR)
        ct_sitk = sitk.ReadImage(str(Path(ROOT) / pid / "image" / "ct.mha"))  # resample reference
        full_shape = ct.array.shape
        if a.real_ct:   # DIAGNOSTIC: density from REAL CT (no synth, no resample) -> should reproduce Task-3 ~97
            from doserad.physics.density import hu_to_density
            density = hu_to_density(ct.array, hu_anchors).astype(np.float32)
        else:
            density, _ = _make_sct_density(synth, scfg, pid, ct_sitk, hu_anchors, dev)
        assert density.shape == full_shape, f"{pid}: density {density.shape} != grid {full_shape}"
        # skin-entry engine gates per-ray internally (no mask_air/body_mask arg); else keep old masking
        mask_kw = {} if (a.unmasked or skin_entry) else dict(body_mask=binary_fill_holes(density >= 0.1), mask_air=True)
        plan = json.load(open(Path(ROOT) / pid / f"{pid}.json"))
        rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]):
                (r["ray_source"], r["ray_target"], bl["energy"])
                for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
        files = sorted(f for f in (cache / pid).glob("B*_R*_L*.npz") if ".tmp" not in f.name)
        pred_cps, gt_cps = [], []
        for f in files:
            z = np.load(f); ch = z["channels"].astype(np.float32); bb = tuple(int(v) for v in z["bbox"])
            z0, z1, y0, y1, x0, x1 = bb
            sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
            b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
            src, tgt, e = rays[(b, r, l)]
            dens_c = density[sl]                                                  # ch0 (sCT density)
            wepl_c = _wepl_on_density(ct, density, src, tgt, bb, pm, dev)         # ch1 (WEPL on sCT)
            pb = pb_fn(ct, src, tgt, e, out_bbox=bb, machine=pm,
                       density_override=density, device=dev,
                       **mask_kw).astype(np.float32)   # PB prior on sCT (skin-entry gated when skin_entry)
            inp = np.stack([dens_c, wepl_c, pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) \
                / _P_CH_SCALE_PRIOR[:, None, None, None]
            pred_cps.append((_predict(eng, inp.astype(np.float32), dev), bb))
            gt_cps.append((z["dose"].astype(np.float32), bb))
        bmaes = []
        for (pr, _), (gt, _) in zip(pred_cps, gt_cps):
            gm = gt.max()
            if gm <= 0: continue
            m = gt >= 0.1 * gm
            if m.any(): bmaes.append(float(np.abs(pr[m] - gt[m]).mean() / gm))
        beam_mae = float(np.mean(bmaes)) if bmaes else float("nan")
        plan_pred = accumulate_plan(pred_cps, full_shape)
        plan_gt = accumulate_plan(gt_cps, full_shape)
        rx = float(plan_gt.max())
        zz, yy, xx = np.where(plan_gt >= 0.05 * rx); mgn = 4
        sl = (slice(max(int(zz.min()) - mgn, 0), int(zz.max()) + mgn + 1),
              slice(max(int(yy.min()) - mgn, 0), int(yy.max()) + mgn + 1),
              slice(max(int(xx.min()) - mgn, 0), int(xx.max()) + mgn + 1))
        pc, gc = plan_pred[sl], plan_gt[sl]
        g1c, g1m = gamma_array(pc, gc, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        ov = float((g1c[g1m] <= 1.0).mean()) if g1m.any() else float("nan")
        g2 = gamma_pass(pc, gc, ct.spacing, rx, dose_pct=2.0, dta_mm=2.0)
        g3 = gamma_pass(pc, gc, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
        site = "lung" if "THB" in pid else "abdomen"
        rows.append({"patient": pid, "site": site, "plan_g1": ov, "plan_g2": g2, "plan_g3": g3, "beam_mae": beam_mae,
                     "strat_mae": stratified_mae(plan_pred, plan_gt, rx)})
        print(f"  {pid} ({site}): PLAN γ1/1 {ov*100:.1f}%  γ2/2 {g2*100:.1f}%  γ3/3 {g3*100:.1f}%  "
              f"[{len(files)} beamlets, {time.time()-t0:.0f}s]", flush=True)
        if a.save_pred:
            sp = Path(a.save_pred); sp.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sp / f"{pid}.npz", pred=plan_pred.astype(np.float32),
                                gt=plan_gt.astype(np.float32), ct=ct.array.astype(np.float32),
                                spacing=np.asarray(ct.spacing, np.float32), rx=np.float32(rx))

    out = Path("/home/kaiwang/doserad2026_workdir/runs") / f"protonmri_plan_{a.label}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    g = lambda s: st.mean([r["plan_g1"] for r in rows if (s is None or r["site"] == s)] or [float("nan")])
    print(f"\n{a.label}: PLAN γ1/1 ALL {g(None)*100:.1f} (abd {g('abdomen')*100:.1f} / lung {g('lung')*100:.1f}) "
          f"| γ2/2 ALL {st.mean([r['plan_g2'] for r in rows])*100:.1f} "
          f"| γ3/3 ALL {st.mean([r['plan_g3'] for r in rows])*100:.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
