"""Localise WHERE the sCT costs proton-MRI its gamma — diagnosis, not a proposed fix.

Motivation: the same proton dose net scores 97.0 on the REAL CT and 83.3 on the sCT
(`runs/eval_protonmri_realct_diag.log`), so 13.7 gamma points live entirely in the density volume.
The obvious remedy (bone-weighting the sCT anchor) is already tested and does nothing (+0.048, p=0.57),
so the next step is to find out where the density error actually is instead of proposing another
physically-plausible tweak.

Two views, both per patient, joined against that patient's plan gamma:

  1. Density error by tissue class, classes defined on the REAL CT (air / lung / soft / bone).
     Reports signed bias and absolute error, plus each class's share of the total error mass.
     Signed bias matters more than |error| for protons: a systematic density offset scales range.

  2. Range error along the ACTUAL beam central axes. For every beam in the plan, integrate density
     from ray_source to ray_target for both volumes; the difference is a water-equivalent path
     length error, reported in mm. This is the quantity that moves the Bragg peak, so it is the one
     that should predict gamma if range is the failure mode.

Per-patient rows are appended to the CSV as they finish, so an interrupted run keeps its work.

  python scripts/diag_sct_error.py \
      --e2e-ckpt  $RUNS/proton_mri_e2e_densdirect/best.pt \
      --config    configs/experiments/proton_mri_e2e_densdirect.yaml \
      --gamma-csv $RUNS/protonmri_plan_protonmri_densdirect.csv \
      --out       $RUNS/diag_sct_error.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from doserad.io.mha import load_mha                       # noqa: E402
from doserad.physics.density import hu_to_density         # noqa: E402
from doserad.physics.machine import load_photon_machine   # noqa: E402
from eval_protonmri_plan import _make_sct_density, ROOT, PHOTON_ROOT, MACHINE   # noqa: E402
from train_dose_e2e import E2E                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--e2e-ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--gamma-csv", default=None, help="per-patient plan gamma to correlate against")
ap.add_argument("--out", required=True)
ap.add_argument("--max-patients", type=int, default=None)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = yaml.safe_load(open(a.config))
anchors = load_photon_machine(MACHINE).hu_anchors

e2e = E2E(cfg).to(dev).eval()
sd = torch.load(a.e2e_ckpt, map_location=dev)
e2e.load_state_dict(sd.get("ema", sd.get("model")))
print(f"loaded {a.e2e_ckpt} (step {sd.get('step','?')}), density_direct={cfg.get('density_direct')}",
      flush=True)

val = json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]
if a.max_patients:
    val = val[:a.max_patients]

out = Path(a.out)
done = set()
if out.exists():
    done = {ln.split(",")[0] for ln in out.read_text().splitlines()[1:] if ln.strip()}
    print(f"resuming: {len(done)} patients already in {out}", flush=True)
else:
    out.write_text("patient,site,n_body,"
                   "bias_air,bias_lung,bias_soft,bias_bone,"
                   "mae_air,mae_lung,mae_soft,mae_bone,"
                   "share_air,share_lung,share_soft,share_bone,"
                   "bias_body,mae_body,range_err_mm_mean,range_err_mm_absmean,n_beams,"
                   "rng_air,rng_lung,rng_soft,rng_bone,"
                   "rngabs_air,rngabs_lung,rngabs_soft,rngabs_bone\n")


def ray_class_decomp(d_sct, d_real, hu, body, spacing, origin, src, tgt, n=512):
    """Split ONE ray's water-equivalent path error by the tissue it passes through.

    The patient-level correlations are underpowered (n=16) and confounded by site, so instead of
    asking "does bone error predict gamma", ask the mechanical question directly: of the mm of range
    error on this ray, how many come from bone samples, from soft, from lung? Classes are taken from
    the REAL CT at each sample point, so the partition does not move with the model.
    Returns (total_mm, {class: mm}).
    """
    src = np.asarray(src, np.float64); tgt = np.asarray(tgt, np.float64)
    ts = np.linspace(0.0, 1.0, n)
    pts = src[None, :] + ts[:, None] * (tgt - src)[None, :]
    idx = (pts - np.asarray(origin, np.float64)[None, :]) / np.asarray(spacing, np.float64)[None, :]
    i = np.rint(idx).astype(np.int64)
    d, h, w = d_sct.shape
    ok = ((i[:, 0] >= 0) & (i[:, 0] < w) & (i[:, 1] >= 0) & (i[:, 1] < h) &
          (i[:, 2] >= 0) & (i[:, 2] < d))
    step_mm = float(np.linalg.norm(tgt - src)) / (n - 1)
    zz, yy, xx = i[ok, 2], i[ok, 1], i[ok, 0]
    de = (d_sct[zz, yy, xx].astype(np.float64) - d_real[zz, yy, xx].astype(np.float64)) * step_mm
    hh, bb = hu[zz, yy, xx], body[zz, yy, xx]
    m = {"air": ~bb, "lung": bb & (hh < -300), "soft": bb & (hh >= -300) & (hh <= 200),
         "bone": bb & (hh > 200)}
    return float(de.sum()), {c: float(de[k].sum()) for c, k in m.items()}


def integrate_density(vol, spacing, origin, src, tgt, n=512):
    """Water-equivalent path length (mm) from src to tgt, both in world mm. Trilinear-free nearest
    sampling is enough here: we compare two volumes on the SAME grid with the SAME sample points, so
    interpolation error cancels in the difference."""
    src = np.asarray(src, np.float64); tgt = np.asarray(tgt, np.float64)
    ts = np.linspace(0.0, 1.0, n)
    pts = src[None, :] + ts[:, None] * (tgt - src)[None, :]          # (n,3) world xyz
    idx = (pts - np.asarray(origin, np.float64)[None, :]) / np.asarray(spacing, np.float64)[None, :]
    i = np.rint(idx).astype(np.int64)                                 # (n,3) -> x,y,z
    d, h, w = vol.shape
    ok = ((i[:, 0] >= 0) & (i[:, 0] < w) & (i[:, 1] >= 0) & (i[:, 1] < h) &
          (i[:, 2] >= 0) & (i[:, 2] < d))
    rho = np.zeros(n, np.float64)
    rho[ok] = vol[i[ok, 2], i[ok, 1], i[ok, 0]]
    step_mm = float(np.linalg.norm(tgt - src)) / (n - 1)
    return float(rho.sum() * step_mm)                                 # g/cm^3 * mm == mm w.e.


gam = {}
if a.gamma_csv and Path(a.gamma_csv).exists():
    import csv
    with open(a.gamma_csv) as fh:
        for r in csv.DictReader(fh):
            gam[r["patient"]] = (r.get("site", ""), float(r["plan_g1"]))

for k, pid in enumerate(val, 1):
    if pid in done:
        continue
    t0 = time.time()
    ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
    ct_sitk = sitk.ReadImage(str(Path(ROOT) / pid / "image" / "ct.mha"))
    hu = ct.array.astype(np.float32)
    d_real = hu_to_density(hu, anchors).astype(np.float32)
    with torch.no_grad():
        d_sct, _ = _make_sct_density(e2e, cfg, pid, ct_sitk, anchors, dev)
    err = (d_sct - d_real).astype(np.float32)

    # classes from the REAL CT, so the partition never moves with the model being diagnosed
    body = d_real >= 0.1
    cls = {"air": ~body,
           "lung": body & (hu < -300),
           "soft": body & (hu >= -300) & (hu <= 200),
           "bone": body & (hu > 200)}
    tot_abs = float(np.abs(err).sum()) or 1.0
    bias = {c: float(err[m].mean()) if m.any() else 0.0 for c, m in cls.items()}
    mae = {c: float(np.abs(err[m]).mean()) if m.any() else 0.0 for c, m in cls.items()}
    share = {c: float(np.abs(err[m]).sum()) / tot_abs for c, m in cls.items()}

    # range error along the real beam central axes
    plan = json.load(open(Path(ROOT) / pid / f"{pid}.json"))
    sp, org = ct.spacing, ct.origin       # (x,y,z) mm
    rerr, dec = [], {"air": [], "lung": [], "soft": [], "bone": []}
    for b in plan["beams"]:
        r = b["rays"][len(b["rays"]) // 2]                 # central ray of the beam
        tot, parts = ray_class_decomp(d_sct, d_real, hu, body, sp, org,
                                      r["ray_source"], r["ray_target"])
        rerr.append(tot)
        for c, v in parts.items():
            dec[c].append(v)
    rerr = np.asarray(rerr, np.float64) if rerr else np.zeros(1)
    dec = {c: np.asarray(v, np.float64) if v else np.zeros(1) for c, v in dec.items()}

    site, g = gam.get(pid, ("", float("nan")))
    with open(out, "a") as fh:
        fh.write(f"{pid},{site},{int(body.sum())},"
                 f"{bias['air']:.5f},{bias['lung']:.5f},{bias['soft']:.5f},{bias['bone']:.5f},"
                 f"{mae['air']:.5f},{mae['lung']:.5f},{mae['soft']:.5f},{mae['bone']:.5f},"
                 f"{share['air']:.4f},{share['lung']:.4f},{share['soft']:.4f},{share['bone']:.4f},"
                 f"{float(err[body].mean()):.5f},{float(np.abs(err[body]).mean()):.5f},"
                 f"{rerr.mean():.3f},{np.abs(rerr).mean():.3f},{len(rerr)},"
                 f"{dec['air'].mean():.3f},{dec['lung'].mean():.3f},"
                 f"{dec['soft'].mean():.3f},{dec['bone'].mean():.3f},"
                 f"{np.abs(dec['air']).mean():.3f},{np.abs(dec['lung']).mean():.3f},"
                 f"{np.abs(dec['soft']).mean():.3f},{np.abs(dec['bone']).mean():.3f}\n")
    print(f"[{k}/{len(val)}] {pid} ({site}) γ {g*100 if g == g else float('nan'):.1f} | "
          f"body bias {float(err[body].mean()):+.4f} g/cm3 | "
          f"bone bias {bias['bone']:+.4f} (share {share['bone']*100:.0f}%) | "
          f"lung bias {bias['lung']:+.4f} (share {share['lung']*100:.0f}%) | "
          f"range err {rerr.mean():+.2f} mm (|.| {np.abs(rerr).mean():.2f}) | {time.time()-t0:.0f}s",
          flush=True)
    del d_sct, d_real, err
    torch.cuda.empty_cache()

print(f"\ndone -> {out}", flush=True)
