"""High-dose-weighted L1 for absolute (scaled) per-CP dose. Voxels in the
high-dose region (gt >= hi_frac * per-sample max) get weight `hi_w`, others
weight 1 — aligning the loss with the challenge's masked-MAE metric, which masks
voxels >= 10% of that beam's maximum GT dose. The threshold is per-sample
relative (not a fixed absolute value) so it tracks each beam's own max."""
from __future__ import annotations

import torch


def weighted_l1(pred: torch.Tensor, gt: torch.Tensor,
                hi_frac: float = 0.1, hi_w: float = 10.0,
                grad_w: float = 0.0, het_w: float = 0.0, lung_w: float = 0.0,
                density: torch.Tensor | None = None,
                lung_lo: float = 0.05, lung_hi: float = 0.6) -> torch.Tensor:
    """High-dose-weighted L1 + optional bounded weights on the CLINICALLY-HARDEST
    dose-calc regions (see [[dose-hard-regions]]):
      - `grad_w` PENUMBRA: ∝ normalized GT dose-gradient |∇dose| (steep field edges);
      - `het_w`  INTERFACES: ∝ normalized density-gradient |∇density| (bone/soft-tissue/
                 lung boundaries — sharp density discontinuities);
      - `lung_w` LUNG: low-density tissue (density in [lung_lo, lung_hi] g/cm³).
    `het_w`/`lung_w` need `density` (raw g/cm³, B,1,D,H,W). All weights are BOUNDED and
    derived from constant targets (gt/density) → no gamma_proxy instability. Defaults 0 →
    identical to the plain high-dose-weighted L1."""
    # gt: (B, 1, D, H, W) — max over spatial dims, per sample.
    sdim = tuple(range(1, gt.ndim))
    smax = gt.amax(dim=sdim, keepdim=True)
    hi = (gt >= hi_frac * smax) & (gt > 0)
    w = torch.where(hi, torch.full_like(gt, hi_w), torch.ones_like(gt))
    if grad_w > 0:
        gz, gy, gx = torch.gradient(gt, dim=(2, 3, 4))
        g = torch.sqrt(gz * gz + gy * gy + gx * gx)
        gmax = g.amax(dim=sdim, keepdim=True).clamp_min(1e-9)
        w = w + grad_w * (g / gmax)
    if density is not None and het_w > 0:
        dz, dy, dx = torch.gradient(density, dim=(2, 3, 4))
        dg = torch.sqrt(dz * dz + dy * dy + dx * dx)
        dgmax = dg.amax(dim=sdim, keepdim=True).clamp_min(1e-9)
        w = w + het_w * (dg / dgmax)
    if density is not None and lung_w > 0:
        lung = ((density > lung_lo) & (density < lung_hi)).to(w.dtype)
        w = w + lung_w * lung
    return (w * (pred - gt).abs()).sum() / w.sum().clamp_min(1.0)


def gamma_proxy(pred: torch.Tensor, gt: torch.Tensor, spacing_mm=2.0,
                dose_pct: float = 0.01, dist_mm: float = 1.0,
                mask_frac: float = 0.1, max_gamma: float = 2.0) -> torch.Tensor:
    """Differentiable, BOUNDED local-gamma surrogate (mirrors the scored 1%/1mm γ).

    First-order DTA approximation: minimizing the gamma functional over a spatial
    shift, to first order in the reference-dose gradient g=|∇D_ref|, gives
        γ(r) ≈ |D_pred − D_ref| / sqrt((dose_pct·D_ref)² + (dist_mm·g)²).
    The denominator (built from GT only) is detached, so gradient flows through the
    numerator |pred−gt|. Scale-invariant: a global dose rescale cancels top/bottom.
    Penalizes only failing voxels (γ>1) inside the ≥mask_frac·beam-max region.

    NUMERIC SAFETY (the v1 run collapsed without these):
      - the local-dose reference is floored at mask_frac·beam-max so the %-criterion
        denominator can't approach 0 in low-gradient high-dose plateaus (which made γ
        explode to ~1e3 and dominate L1 ~1e4×);
      - γ is CAPPED at `max_gamma`, so relu(γ-1)² ∈ [0,(max_gamma-1)²] and the loss is
        bounded — no fp16 gradient blow-up.
    Returns 0 if the mask is empty. pred/gt: (B,1,D,H,W).
    """
    if isinstance(spacing_mm, (int, float)):
        sp = (float(spacing_mm),) * 3
    else:
        sp = tuple(float(s) for s in spacing_mm)
    gz, gy, gx = torch.gradient(gt, spacing=sp, dim=(2, 3, 4))
    g = torch.sqrt(gz * gz + gy * gy + gx * gx)
    smax = gt.amax(dim=tuple(range(1, gt.ndim)), keepdim=True)
    gt_ref = torch.clamp_min(gt, mask_frac * smax)          # floor the local %-reference
    dd_crit = dose_pct * gt_ref
    denom = torch.sqrt(dd_crit * dd_crit + (dist_mm * g) ** 2).clamp_min(1e-9).detach()
    gamma = ((pred - gt).abs() / denom).clamp(max=max_gamma)  # CAP -> bounded loss
    mask = (gt >= mask_frac * smax) & (gt > 0)
    if not bool(mask.any()):
        return pred.new_zeros(())
    fail = torch.relu(gamma - 1.0) ** 2
    return fail[mask].mean()
