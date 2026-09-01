"""Quantify whether the beam-bbox crop (fluence>0 + 16mm margin) covers the
photon scatter tail down to the gamma floor (>=10% beam-max).

For each cached CP crop we ask: does >=10%-beam-max dose reach the crop boundary?
If yes, the scatter tail is being clipped -> unrecoverable miss in the
accumulated plan (bbox-outside prediction is always 0).

Reports, over a sample of crops:
  - frac_clipped: fraction of crops whose 1-voxel boundary shell holds dose >= 10% beam-max
  - boundary_ratio: (max dose on boundary shell)/(beam-max), distribution
  - edge_mass: fraction of >=10% dose mass sitting within 2 voxels of the boundary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon")


def boundary_shell_max(d: np.ndarray) -> float:
    faces = [d[0], d[-1], d[:, 0], d[:, -1], d[:, :, 0], d[:, :, -1]]
    return max(float(f.max()) for f in faces)


def edge_mass_frac(d: np.ndarray, thr: float, k: int = 2) -> float:
    m = d >= thr
    if not m.any():
        return 0.0
    edge = np.zeros_like(m)
    edge[:k] = edge[-k:] = True
    edge[:, :k] = edge[:, -k:] = True
    edge[:, :, :k] = edge[:, :, -k:] = True
    hi = d[m].sum()
    return float(d[m & edge].sum() / hi) if hi > 0 else 0.0


def main():
    n_per_patient = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pdirs = sorted([p for p in CACHE.iterdir() if p.is_dir()])
    ratios = []
    clipped = 0
    edge_masses = []
    n = 0
    for pd in pdirs:
        files = sorted(pd.glob("*.npz"))
        if not files:
            continue
        pick = files[:: max(1, len(files) // n_per_patient)][:n_per_patient]
        for f in pick:
            z = np.load(f)
            d = z["dose"].astype(np.float32)
            bmax = float(d.max())
            if bmax <= 0:
                continue
            thr = 0.10 * bmax
            bsm = boundary_shell_max(d)
            r = bsm / bmax
            ratios.append(r)
            if bsm >= thr:
                clipped += 1
            edge_masses.append(edge_mass_frac(d, thr))
            n += 1
    ratios = np.array(ratios)
    em = np.array(edge_masses)
    print(f"crops sampled: {n}  (from {len(pdirs)} patients, ~{n_per_patient}/patient)")
    print(f"frac_clipped (>=10% dose touches boundary): {clipped/n*100:.1f}%")
    print(f"boundary_ratio (max-edge-dose / beam-max):")
    print(f"   median {np.median(ratios)*100:.1f}%  p90 {np.percentile(ratios,90)*100:.1f}%  "
          f"p99 {np.percentile(ratios,99)*100:.1f}%  max {ratios.max()*100:.1f}%")
    print(f"edge_mass (>=10% dose mass within 2vox of boundary): "
          f"median {np.median(em)*100:.2f}%  p90 {np.percentile(em,90)*100:.2f}%  "
          f"max {em.max()*100:.2f}%")


if __name__ == "__main__":
    main()
