"""First-order analytical 'naive dose' from the cached 5 physics channels, to be
used as a strong prior input channel (DeepDose-style residual learning). Computed
on-the-fly from existing cache — NO 215 GB recompute.

naive ≈ energy_fluence · inverse_square · buildup
  - fluence channel  = open_mask·exp(-μ·rdepth)  (already has primary attenuation falloff)
  - inverse_square   = (SAD/source_dist)²        (geometric divergence; missing from fluence)
  - buildup(rdepth)  = electron buildup ramp (low surface dose → 1 beyond d_max ~1.5 g/cm²)

Channel order in the cache: [density, rdepth, fluence, dist_to_cax, source_dist].
"""
from __future__ import annotations

import numpy as np

SAD_MM = 1000.0
_SURFACE = 0.2          # ~surface dose fraction at 6 MV
_TAU = 0.5             # buildup scale (g/cm²)

# --- v10-lite density-scaled lateral SCATTER (Tier-1.5, O'Connor-inspired) ---
# The Tier-1 prior is scatter-blind → fails in lung. Add a density-scaled blur of
# the primary: scatter spreads MORE in low density (kernel width ∝ 1/density). We
# approximate the spatially-varying width by blending two fixed-σ Gaussians by a
# local "lung-ness" factor α(density). σ in VOXELS (2mm dose grid).
_SCAT_SIGMA_TISSUE = 2.0   # ~4 mm lateral scatter in soft tissue
_SCAT_SIGMA_LUNG = 8.0     # ~16 mm in lung (σ ∝ 1/ρ, ρ_lung~0.25 → ~4× tissue)
_RHO_LUNG = 0.3            # density at/below which → full wide kernel
_RHO_SOFT = 0.9            # density at/above which → narrow kernel


def compute_naive_dose(channels: np.ndarray, scatter: bool = False,
                       skin_gate: bool = False) -> np.ndarray:
    """channels: (5, ...) cached stack. Returns naive dose (...), same spatial shape.

    `scatter=False` (default): Tier-1 primary = fluence·inv_sq·buildup (UNCHANGED;
    used by v6/v8/v9). `scatter=True`: v10-lite Tier-1.5 — convolve the primary with
    a density-scaled lateral-scatter kernel (wide in lung, narrow in tissue) to supply
    the lateral spread the Tier-1 prior is missing. MUST stay identical at train+infer.

    `skin_gate` (opt-in, default False = OFF = byte-identical to before): apply the
    skin-ENTRY gate — zero the prior ONLY at voxels the beam has not yet reached, i.e.
    the external air UPSTREAM of the skin crossing (skin-entry rdepth == 0 there). This
    removes the spurious build-up-floor dose in the entrance air gap while NEVER masking
    anything in or behind the body (rdepth>0), including internal low-density / air
    cavities — dose inside the patient is left untouched. Same `entered` semantics as the
    proton skin-entry engine and diff_channels_skinentry. Requires skin-entry channels
    (rdepth==0 strictly upstream). MUST be identical at train+infer (thread via config)."""
    rdepth = channels[1].astype(np.float32)        # g/cm²
    fluence = channels[2].astype(np.float32)
    source_dist = channels[4].astype(np.float32)   # mm
    density = channels[0].astype(np.float32)
    inv_sq = (SAD_MM / np.clip(source_dist, 1.0, None)) ** 2
    buildup = 1.0 - (1.0 - _SURFACE) * np.exp(-rdepth / _TAU)
    primary = (fluence * inv_sq * buildup).astype(np.float32)
    if not scatter:
        out = primary
    else:
        # SOFT TISSUE keeps the SHARP primary (blurring it hurt — Stage-A 2026-06-12);
        # only LOW-DENSITY (lung) gets the wide density-scaled scatter spread.
        blur_w = _sep_gauss3d(primary, _SCAT_SIGMA_LUNG)
        alpha = np.clip((_RHO_SOFT - density) / (_RHO_SOFT - _RHO_LUNG), 0.0, 1.0)
        out = ((1.0 - alpha) * primary + alpha * blur_w).astype(np.float32)
    if skin_gate:
        # entered gate: rdepth==0 ONLY in the external air upstream of the skin (skin-entry
        # raytrace zeros pre-skin density). rdepth>0 everywhere in/behind the body, so this
        # never masks in-body voxels (tissue OR internal air cavities).
        out = (out * (rdepth > 0)).astype(np.float32)
    return out


def _sep_gauss3d(vol: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 3D Gaussian blur via torch (CPU) — no scipy dep, and the same
    conv math the GPU inference path will use (train/inference consistency)."""
    import torch
    t = torch.as_tensor(vol, dtype=torch.float32).view(1, 1, *vol.shape)
    r = max(int(3 * sigma), 1)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-(x * x) / (2 * sigma * sigma)); g = g / g.sum()
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[dim] = g.numel()
        pad = [0, 0, 0]; pad[dim - 2] = r
        t = torch.nn.functional.conv3d(t, g.view(shape), padding=tuple(pad))
    return t.view(*vol.shape).numpy()
