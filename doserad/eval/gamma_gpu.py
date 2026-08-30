"""3D local gamma on the GPU, matching the challenge evaluator's pymedphys call.

The evaluator scores with pymedphys (local, 1%/1 mm, 10% cutoff, max_gamma=2, interp_fraction=10)
and so do we -- but pymedphys is single-threaded CPU and takes ~100-200 s for one photon plan
(~870k evaluated voxels). That is now the rate limiter on our own experiments: re-scoring a handful
of checkpoints across the CV folds is hours, and the corrected-evaluation work makes many such
sweeps necessary. It contributes nothing to the submitted runtime -- the organizers compute gamma
on their side -- so this is purely a research-loop tool.

Formulation (identical to the shell method pymedphys implements):

    gamma(x)^2 = min over displacement d of  |d|^2 / dta^2
                                           + (D_eval(x + d) - D_ref(x))^2 / tol(x)^2
    tol(x) = dose_percent_threshold/100 * D_ref(x)        (local normalisation)

Evaluated only where D_ref >= lower_percent_dose_cutoff/100 * max(D_ref), and capped at max_gamma.

Two properties make this a good GPU fit: every voxel is independent, and gamma >= |d|/dta, so
displacements can be swept in order of increasing radius and a voxel retired as soon as its running
minimum drops below (r/dta)^2 -- most voxels pass with gamma < 1 and retire in the first shells.

`gamma_pass_gpu` is a drop-in for `doserad.eval.gamma.gamma_pass`. VALIDATE IT before trusting it:
scripts/validate_gamma_gpu.py compares against pymedphys on real plans.
"""
from __future__ import annotations

import numpy as np
import torch


def _shell_offsets(dta_mm: float, max_gamma: float, step_mm: float):
    """Displacements inside the search sphere, ordered by radius so the sweep can retire voxels
    early. Radius is max_gamma*dta because gamma >= |d|/dta makes anything beyond it unreachable."""
    r_max = max_gamma * dta_mm
    n = int(np.ceil(r_max / step_mm))
    ax = np.arange(-n, n + 1) * step_mm
    dz, dy, dx = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.stack([dz.ravel(), dy.ravel(), dx.ravel()], 1)
    r = np.linalg.norm(d, axis=1)
    keep = r <= r_max + 1e-9
    d, r = d[keep], r[keep]
    order = np.argsort(r)
    return d[order], r[order]


@torch.no_grad()
def gamma_array_gpu(pred: np.ndarray, gt: np.ndarray, spacing, rx: float,
                    dose_pct: float = 1.0, dta_mm: float = 1.0, hi_frac: float = 0.1,
                    max_gamma: float = 2.0, interp_fraction: int = 10,
                    device: str = "cuda", chunk: int = 1 << 20):
    """(gamma, eval_mask) with gamma=inf outside the mask, mirroring doserad.eval.gamma.gamma_array.

    `spacing` is (sx, sy, sz) mm as SimpleITK reports it; arrays are (z, y, x).
    """
    sx, sy, sz = (float(v) for v in spacing)
    gt_t = torch.as_tensor(np.ascontiguousarray(gt, np.float32), device=device)
    pr_t = torch.as_tensor(np.ascontiguousarray(pred, np.float32), device=device)
    nz, ny, nx = gt_t.shape

    mask = gt_t >= hi_frac * rx
    out = torch.full(gt_t.shape, float("inf"), device=device, dtype=torch.float32)
    if not bool(mask.any()):
        return out.cpu().numpy().astype(np.float64), mask.cpu().numpy()

    idx = mask.nonzero(as_tuple=False)                       # (N, 3) in (z, y, x)
    ref = gt_t[mask]                                          # (N,)
    tol = (dose_pct / 100.0) * ref                            # local normalisation
    tol = torch.where(tol > 0, tol, torch.full_like(tol, torch.finfo(torch.float32).eps))

    # physical position of each evaluated voxel, and the grid extent for normalised sampling
    pos = torch.stack([idx[:, 0] * sz, idx[:, 1] * sy, idx[:, 2] * sx], 1)      # (N,3) mm, (z,y,x)
    ext = torch.tensor([(nz - 1) * sz, (ny - 1) * sy, (nx - 1) * sx], device=device)

    offs_np, rad_np = _shell_offsets(dta_mm, max_gamma, dta_mm / interp_fraction)
    offs = torch.as_tensor(offs_np, dtype=torch.float32, device=device)
    rad = torch.as_tensor(rad_np, dtype=torch.float32, device=device)

    vol = pr_t[None, None]                                    # (1,1,Z,Y,X) for grid_sample
    g2 = torch.full((idx.shape[0],), float(max_gamma) ** 2, device=device)

    for s in range(0, idx.shape[0], chunk):
        e = min(s + chunk, idx.shape[0])
        p, r_, t_ = pos[s:e], ref[s:e], tol[s:e]
        g2c = torch.full((e - s,), float(max_gamma) ** 2, device=device)
        for k in range(offs.shape[0]):
            floor = (rad[k] / dta_mm) ** 2
            if floor >= float(max_gamma) ** 2:
                break                                          # nothing further can improve on max
            if bool((g2c <= floor).all()):
                break                                          # every voxel already retired
            q = p + offs[k]                                    # (M,3) mm, (z,y,x)
            # grid_sample wants normalised (x, y, z) in [-1, 1], align_corners=True
            gz = (q[:, 0] / ext[0]) * 2 - 1
            gy = (q[:, 1] / ext[1]) * 2 - 1
            gx = (q[:, 2] / ext[2]) * 2 - 1
            grid = torch.stack([gx, gy, gz], 1).view(1, 1, 1, -1, 3)
            de = torch.nn.functional.grid_sample(
                vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(-1)
            cand = floor + ((de - r_) / t_) ** 2
            g2c = torch.minimum(g2c, cand)
        g2[s:e] = g2c

    out[mask] = torch.sqrt(g2.clamp(max=float(max_gamma) ** 2))
    return out.cpu().numpy().astype(np.float64), mask.cpu().numpy()


def gamma_pass_gpu(pred, gt, spacing, rx, dose_pct=1.0, dta_mm=1.0, hi_frac=0.1, **kw) -> float:
    g, m = gamma_array_gpu(pred, gt, spacing, rx, dose_pct, dta_mm, hi_frac, **kw)
    return float((g[m] <= 1.0).mean()) if m.any() else float("nan")
