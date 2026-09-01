"""Smoke test: AAA BEV-convolution prior on one real lung CP vs GT and naive.
Runs on CPU (no GPU contention). Reports shape/finiteness + LSQ-scaled correlation
with GT, side by side with the Tier-1 naive prior."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doserad.beam.parse import load_photon_plan
from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.machine import load_photon_machine
from doserad.physics.naive_dose import compute_naive_dose
from doserad.physics.priors.pencil_beam_aaa import load_aaa_kernel, aaa_prior_dose

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = "/home/kaiwang/doserad2026_workdir/cache/crops/photon"
PID = sys.argv[1] if len(sys.argv) > 1 else "1THB002"
DEV = os.environ.get("SMOKE_DEV", "cpu")


def lsq_corr(pred, gt, mask):
    p = pred[mask].astype(np.float64); g = gt[mask].astype(np.float64)
    a = float((p * g).sum() / max((p * p).sum(), 1e-12))   # LSQ scale pred->gt
    r = float(np.corrcoef(p, g)[0, 1])
    rel_mae = float(np.abs(a * p - g).mean() / max(g.mean(), 1e-12))
    return a, r, rel_mae


def main():
    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    plan = load_photon_plan(f"{ROOT}/{PID}/{PID}.json")
    ct = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    density = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
    kernel = load_aaa_kernel()
    print(f"patient {PID}  shape {ct.array.shape}  spacing {ct.spacing}")

    cdir = f"{CACHE}/{PID}"
    beam = plan.beams[0]
    cp = beam.control_points[len(beam.control_points) // 2]
    f = f"{cdir}/{beam.beam_idx}_{cp.cp_idx:03d}.npz"
    if not os.path.exists(f):
        # fall back to first existing cache file
        import glob
        files = sorted(glob.glob(f"{cdir}/*.npz"))
        f = files[len(files) // 2]
    z = np.load(f)
    gt = z["dose"].astype(np.float32)
    bbox = tuple(int(v) for v in z["bbox"])
    channels = z["channels"]
    print(f"CP cache {os.path.basename(f)}  gt shape {gt.shape}  bbox {bbox}  gt[max]={gt.max():.4g}")

    # match the CP this cache file came from
    bidx, cidx = (int(x) for x in os.path.basename(f)[:-4].split("_"))
    beam = next(b for b in plan.beams if b.beam_idx == bidx)
    cp = next(c for c in beam.control_points if c.cp_idx == cidx)
    iso = np.asarray(beam.iso_center, dtype=np.float64)
    mid = beam.control_points[len(beam.control_points) // 2].gantry_angle
    src = beam_source_pos(iso, machine.sad_mm, cp.gantry_angle)
    axis, u_hat, v_hat = beam_basis(cp.gantry_angle)

    import time
    t0 = time.time()
    aaa = aaa_prior_dose(density, ct.spacing, ct.origin, src, axis, u_hat, v_hat, iso,
                         machine, np.asarray(cp.mlc_left_int_mm),
                         np.asarray(cp.mlc_right_int_mm), kernel,
                         out_bbox=bbox, device=DEV)
    dt = time.time() - t0
    naive = compute_naive_dose(channels)

    print(f"\naaa shape {aaa.shape}  finite={np.isfinite(aaa).all()}  "
          f"min={aaa.min():.4g} max={aaa.max():.4g}  ({dt:.2f}s on {DEV})")
    assert aaa.shape == gt.shape, "shape mismatch vs GT"

    mask = gt > 0.05 * gt.max()
    a_a, r_a, m_a = lsq_corr(aaa, gt, mask)
    a_n, r_n, m_n = lsq_corr(naive, gt, mask)
    print(f"\n            corr    rel-MAE(LSQ)   scale")
    print(f"  naive :  {r_n:+.4f}   {m_n*100:6.2f}%   {a_n:.4g}")
    print(f"  AAA   :  {r_a:+.4f}   {m_a*100:6.2f}%   {a_a:.4g}")
    print(f"\n  {'AAA BEATS naive' if r_a > r_n else 'naive still better'} on corr; "
          f"{'AAA BEATS naive' if m_a < m_n else 'naive still better'} on rel-MAE")


if __name__ == "__main__":
    main()
