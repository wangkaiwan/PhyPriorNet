"""Distal RANGE (R80/R90) accuracy of predicted proton dose vs MC GT — a proton-range
discriminator to back the manuscript's proton claims (gamma alone is a weak range metric).

For each beamlet we sample the 1-D central-axis depth-dose profile of BOTH pred and GT by
trilinear-interpolating the beamlet crop along the beam central axis (the ray_source->ray_target
line, extended into the patient). The distal R80 (R90) is the DEEPEST position where the profile
crosses 80% (90%) of its own peak on the falling edge. Range error = |R80_pred - R80_GT| (mm).
Aggregated per patient (mean over beamlets), then mean +/- SD across fold-0 val patients, split by
task (Proton-CT / Proton-MRI) and site (abdomen / lung).

Non-destructive: reuses (imports) the forward/channel helpers from eval_proton_plan.py and
eval_protonmri_plan.py; does NOT modify them.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/proton_range_metric.py \
      --task ct  --topk 100 [--verify]
  CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/proton_range_metric.py \
      --task mri --topk 100
"""
from __future__ import annotations
import argparse, json, sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import statistics as st
from pathlib import Path
import numpy as np
import torch
import yaml

from doserad.io.mha import load_mha

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
CT_CFG = "configs/experiments/cv/ft_skinentry_protonct_f0.yaml"
CT_CKPT = "/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt"
MRI_CFG = "configs/experiments/cv/se_protonmri_f0.yaml"
MRI_CKPT = "/home/kaiwang/doserad2026_workdir/runs/se_protonmri_f0/best.pt"

STEP_MM = 0.5    # depth sampling step along central axis


# ---------------------------------------------------------------------------
# central-axis depth-dose profile + distal range extraction
# ---------------------------------------------------------------------------
def _trilinear(vol, iz, iy, ix):
    """Sample vol[(z,y,x)] at fractional indices (arrays). Points outside -> 0."""
    dz, dy, dx = vol.shape
    z0 = np.floor(iz).astype(int); y0 = np.floor(iy).astype(int); x0 = np.floor(ix).astype(int)
    fz = iz - z0; fy = iy - y0; fx = ix - x0
    out = np.zeros(iz.shape, np.float64)
    for ddz in (0, 1):
        for ddy in (0, 1):
            for ddx in (0, 1):
                zz = z0 + ddz; yy = y0 + ddy; xx = x0 + ddx
                inb = (zz >= 0) & (zz < dz) & (yy >= 0) & (yy < dy) & (xx >= 0) & (xx < dx)
                wz = fz if ddz else (1 - fz); wy = fy if ddy else (1 - fy); wx = fx if ddx else (1 - fx)
                w = wz * wy * wx
                idx = np.where(inb)
                out[idx] += w[idx] * vol[zz[idx], yy[idx], xx[idx]]
    return out


def depth_profile(crop, bbox, source, axis_unit, spacing, origin, step=STEP_MM):
    """1-D central-axis depth-dose. Returns (t_mm, dose) where t is signed depth along axis_unit
    measured from `source` (world). Sampling spans exactly the crop's axial extent."""
    z0, z1, y0, y1, x0, x1 = bbox
    sx, sy, sz = spacing; ox, oy, oz = origin
    # crop-bbox 8 corners in world coords -> project onto axis to get depth span
    zc = np.array([z0, z1]); yc = np.array([y0, y1]); xc = np.array([x0, x1])
    ts = []
    for zi in zc:
        for yi in yc:
            for xi in xc:
                w = np.array([ox + xi * sx, oy + yi * sy, oz + zi * sz]) - source
                ts.append(float(np.dot(w, axis_unit)))
    tmin, tmax = min(ts), max(ts)
    t = np.arange(tmin, tmax + step, step)
    pts = source[None, :] + t[:, None] * axis_unit[None, :]     # (N,3) world x,y,z
    ix = (pts[:, 0] - ox) / sx - x0
    iy = (pts[:, 1] - oy) / sy - y0
    iz = (pts[:, 2] - oz) / sz - z0
    dose = _trilinear(crop, iz, iy, ix)
    return t, dose


def distal_R(t, dose, frac):
    """Deepest position (max t) where dose crosses `frac`*peak on the falling edge, interpolated.
    Returns None if profile has no usable peak."""
    if dose.size < 3:
        return None
    pk = dose.max()
    if pk <= 0:
        return None
    thr = frac * pk
    pk_i = int(np.argmax(dose))
    # walk distally (increasing t / index) from the peak until we drop below thr
    i = pk_i
    while i + 1 < dose.size and dose[i + 1] >= thr:
        i += 1
    if i + 1 >= dose.size:
        # never falls below thr within crop -> distal edge not captured
        return None
    d_hi, d_lo = dose[i], dose[i + 1]            # d_hi >= thr > d_lo
    if d_hi == d_lo:
        return float(t[i])
    frac_step = (d_hi - thr) / (d_hi - d_lo)
    return float(t[i] + frac_step * (t[i + 1] - t[i]))


def beamlet_range_errors(pred, gt, bbox, source, axis_unit, spacing, origin):
    """Returns (dR80, dR90, ok) in mm; ok False if either profile lacks a captured distal edge."""
    tg, dg = depth_profile(gt, bbox, source, axis_unit, spacing, origin)
    tp, dp = depth_profile(pred, bbox, source, axis_unit, spacing, origin)
    g80 = distal_R(tg, dg, 0.80); p80 = distal_R(tp, dp, 0.80)
    g90 = distal_R(tg, dg, 0.90); p90 = distal_R(tp, dp, 0.90)
    if g80 is None or p80 is None:
        return None, None, None, False
    s80 = p80 - g80                       # SIGNED: + = prediction overshoots (deeper than GT)
    d80 = abs(s80)
    d90 = abs(p90 - g90) if (g90 is not None and p90 is not None) else None
    return d80, d90, s80, True


def axis_from_ray(src_world, tgt_world):
    """Central-axis unit vector (into patient) and the machine source point used as depth origin.
    Depth is measured from ray_source; deeper = larger t."""
    src = np.asarray(src_world, np.float64); tgt = np.asarray(tgt_world, np.float64)
    axis = tgt - src; axis = axis / (np.linalg.norm(axis) + 1e-12)
    return src, axis


# ---------------------------------------------------------------------------
# Proton-CT
# ---------------------------------------------------------------------------
def run_ct(topk, verify, patients=None):
    import torch.nn.functional as F
    from doserad.model.unet3d import DoseUNet3D
    from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
    from eval_proton_plan import _predict
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(CT_CFG))
    cache = Path(cfg["cache_dir"]); prior = Path(cfg["prior_dir"]); wepl_dir = Path(cfg["wepl_dir"])
    val = json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]
    if patients:
        val = [p for p in val if p in patients]
    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(dev).eval()
    sd = torch.load(CT_CKPT, map_location=dev); net.load_state_dict(sd.get("ema", sd.get("model")))
    return _run(val, cache, dev, topk, verify, "ct",
                predict_fn=lambda pid, f, ct: _ct_pred(net, dev, cache, prior, wepl_dir, pid, f,
                                                        PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR, _predict))


def _ct_pred(net, dev, cache, prior, wepl_dir, pid, f, DSCALE, PCS, _predict):
    z = np.load(cache / pid / f); ch = z["channels"].astype(np.float32)
    bb = tuple(int(v) for v in z["bbox"])
    ch[1] = np.load(wepl_dir / pid / f)["wepl"].astype(np.float32)
    pb = np.load(prior / pid / f)["pb_prior"].astype(np.float32)
    inp = np.stack([ch[0], ch[1], pb * DSCALE, ch[2], ch[3]], 0) / PCS[:, None, None, None]
    pred = _predict(net, inp.astype(np.float32), dev)
    return pred, z["dose"].astype(np.float32), bb


# ---------------------------------------------------------------------------
# Proton-MRI (E2E dose-aware, MRI->sCT density->proton dose)
# ---------------------------------------------------------------------------
def run_mri(topk, verify, patients=None):
    import SimpleITK as sitk
    from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
    from doserad.physics.machine import load_photon_machine
    from doserad.physics.proton_pb_gpu import ProtonMachineData
    from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry
    from train_dose_e2e import E2E
    from eval_protonmri_plan import _wepl_on_density, _predict, _make_sct_density
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(MRI_CFG))
    cache = Path(cfg["cache_dir"])
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    hu_anchors = machine.hu_anchors
    pm = ProtonMachineData(device=dev)
    val = json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]
    if patients:
        val = [p for p in val if p in patients]
    e2e = E2E(cfg).to(dev).eval()
    sd = torch.load(MRI_CKPT, map_location=dev); e2e.load_state_dict(sd.get("ema", sd.get("model")))
    eng = e2e.dose

    state = {}

    def predict_fn(pid, f, ct):
        if state.get("pid") != pid:
            ct_sitk = sitk.ReadImage(str(Path(ROOT) / pid / "image" / "ct.mha"))
            density, _ = _make_sct_density(e2e, cfg, pid, ct_sitk, hu_anchors, dev)
            plan = json.load(open(Path(ROOT) / pid / f"{pid}.json"))
            rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]):
                    (r["ray_source"], r["ray_target"], bl["energy"])
                    for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
            state.clear(); state.update(pid=pid, density=density, rays=rays)
        density = state["density"]; rays = state["rays"]
        z = np.load(cache / pid / f); ch = z["channels"].astype(np.float32)
        bb = tuple(int(v) for v in z["bbox"]); z0, z1, y0, y1, x0, x1 = bb
        sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
        b, r, l = (int(f.split("_")[i][1:].replace(".npz", "")) for i in range(3))
        src, tgt, e = rays[(b, r, l)]
        dens_c = density[sl]
        wepl_c = _wepl_on_density(ct, density, src, tgt, bb, pm, dev)
        pb = proton_pb_dose_gpu_skinentry(ct, src, tgt, e, out_bbox=bb, machine=pm,
                                          density_override=density, device=dev).astype(np.float32)
        inp = np.stack([dens_c, wepl_c, pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) \
            / _P_CH_SCALE_PRIOR[:, None, None, None]
        pred = _predict(eng, inp.astype(np.float32), dev)
        return pred, z["dose"].astype(np.float32), bb

    return _run(val, cache, dev, topk, verify, "mri", predict_fn=predict_fn)


# ---------------------------------------------------------------------------
# shared driver
# ---------------------------------------------------------------------------
def _select_beamlets(cache, pid, topk):
    files = sorted(f for f in (cache / pid).glob("B*_R*_L*.npz") if ".tmp" not in f.name)
    dmax = []
    for f in files:
        try:
            dmax.append((float(np.load(f)["dose_max"]), f.name))
        except Exception:
            dmax.append((float(np.load(f)["dose"].max()), f.name))
    dmax.sort(reverse=True)
    sel = [n for _, n in dmax[:topk]]
    return sel


def _load_rays(pid):
    plan = json.load(open(Path(ROOT) / pid / f"{pid}.json"))
    return {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]):
            (r["ray_source"], r["ray_target"]) for b in plan["beams"]
            for r in b["rays"] for bl in r["beamlets"]}


def _run(val, cache, dev, topk, verify, task, predict_fn):
    rows = []
    verify_dumps = []
    for pid in val:
        t0 = time.time()
        ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
        spacing = ct.spacing; origin = ct.origin
        rays = _load_rays(pid)
        sel = _select_beamlets(cache, pid, topk)
        d80s, d90s, s80s = [], [], []
        for f in sel:
            b, r, l = (int(f.split("_")[i][1:].replace(".npz", "")) for i in range(3))
            src_w, tgt_w = rays[(b, r, l)]
            source, axis = axis_from_ray(src_w, tgt_w)
            pred, gt, bb = predict_fn(pid, f, ct)
            d80, d90, s80, ok = beamlet_range_errors(pred, gt, bb, source, axis, spacing, origin)
            if not ok:
                continue
            d80s.append(d80); s80s.append(s80)
            if d90 is not None:
                d90s.append(d90)
            if verify and len(verify_dumps) < 6:
                tg, dg = depth_profile(gt, bb, source, axis, spacing, origin)
                tp, dp = depth_profile(pred, bb, source, axis, spacing, origin)
                verify_dumps.append((pid, f, tg, dg, tp, dp, d80))
        site = "lung" if "THB" in pid else "abdomen"
        pm80 = float(np.mean(d80s)) if d80s else float("nan")
        pm90 = float(np.mean(d90s)) if d90s else float("nan")
        sm80 = float(np.mean(s80s)) if s80s else float("nan")
        sd80 = float(np.std(s80s)) if s80s else float("nan")
        rows.append({"patient": pid, "site": site, "n_used": len(d80s), "n_sel": len(sel),
                     "mean_dR80_mm": pm80, "mean_dR90_mm": pm90,
                     "mean_signed_R80_mm": sm80, "std_signed_R80_mm": sd80})
        print(f"  [{task}] {pid} ({site}): dR80 {pm80:.2f}mm  dR90 {pm90:.2f}mm  "
              f"[{len(d80s)}/{len(sel)} beamlets usable, {time.time()-t0:.0f}s]", flush=True)

    if verify:
        _plot_verify(verify_dumps, task)
    return rows


def _plot_verify(dumps, task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(dumps)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, (pid, f, tg, dg, tp, dp, d80) in zip(axes, dumps):
        ax.plot(tg, dg / (dg.max() + 1e-12), label="GT", lw=1.5)
        ax.plot(tp, dp / (dp.max() + 1e-12), label="pred", lw=1.5)
        ax.axhline(0.8, color="k", ls=":", lw=0.7)
        ax.set_title(f"{pid} {f}\ndR80={d80:.2f}mm", fontsize=7)
        ax.set_xlabel("depth along axis (mm)"); ax.legend(fontsize=6)
    fig.tight_layout()
    out = Path("docs/reports") / f"proton_range_verify_{task}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); print(f"[verify] wrote {out}", flush=True)


def summarize(rows, task):
    def agg(site, key):
        vals = [r[key] for r in rows if (site is None or r["site"] == site) and not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0), len(vals)
    print(f"\n==== {task} distal range error (patient-level mean +/- SD) ====")
    for site in ("abdomen", "lung", None):
        m80, s80, n = agg(site, "mean_dR80_mm")
        m90, s90, _ = agg(site, "mean_dR90_mm")
        print(f"  {str(site or 'ALL'):8s} (n={n}): R80 {m80:.2f} +/- {s80:.2f} mm | "
              f"R90 {m90:.2f} +/- {s90:.2f} mm", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ct", "mri"])
    ap.add_argument("--topk", type=int, default=100, help="highest-GT-dose beamlets per patient")
    ap.add_argument("--verify", action="store_true", help="dump 6 example profiles to png and exit-ish")
    ap.add_argument("--patients", default=None, help="comma-list subset (e.g. one patient for --verify)")
    ap.add_argument("--out-csv", default=None)
    a = ap.parse_args()
    pats = set(a.patients.split(",")) if a.patients else None
    fn = run_ct if a.task == "ct" else run_mri
    rows = fn(a.topk, a.verify, patients=pats)
    summarize(rows, a.task)
    out = a.out_csv or f"/home/kaiwang/doserad2026_workdir/runs/proton_range_{a.task}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
