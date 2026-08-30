"""Fast GPU (torch) re-implementation of pyRadPlan/matRad's SVD photon
pencil-beam (AAA-class) dose, for use as a Tier-2 heterogeneity-aware prior.

EXACT model (from pyRadPlan `dose/engines/_svdpb.py::_compute_bixel`), per voxel:

    dose = [ Σ_c  kernel_c(lat) · β_c/(β_c − m) · (e^{−m·rd} − e^{−β_c·rd}) ]
           · (SAD / geo_depth)²

  rd        = radiological (water-equiv) depth along the ray  [our `rdepth` channel]
  geo_depth = geometric source→voxel distance                [our `source_dist` channel]
  lat       = lateral distance from the pencil axis (in the BEV plane)
  kernel_c  = the 3 SVD radial kernel components at the relevant SSD (kernel_data)
  β_c, m    = kernel_betas, m (attenuation)

For a FIELD/CP the lateral term is the OPEN aperture convolved with the radial
kernel in the BEV (beam's-eye-view) plane. Kernel data exported from pyRadPlan:
`data/kernels/photons_Generic_kernel.npz`.

STATUS: kernel loader + EXACT depth-term + 2D radial-kernel builder + BEV-lateral
convolution + patient-grid resampling (`aaa_prior_dose`) = DONE & smoke-validated
(scripts/smoke_aaa.py, 2026-06-13): on lung+abdomen CPs it beats the Tier-1 naive prior
on BOTH LSQ-corr and rel-MAE (e.g. abdomen 1ABB006 corr .961 vs .944, rel-MAE 12.3 vs 14.3%;
lung 1THB016 .781 vs .767, 49.2 vs 55.4%). Warm GPU runtime ~120 ms/CP (cold 504 ms = JIT) →
fits the ≤1s/dosemap gate alongside the 241ms model.
NUMERICAL CHECK vs pyRadPlan _svdpb (scripts/validate_aaa_vs_pyradplan.py, 32 CPs, 2026-06-13):
corr +0.892 ± 0.039, rel-MAE 20.6 ± 6.0 %, gamma1/1 41.2 % — our reimpl captures the right SHAPE
(corr ~0.89) but is an APPROXIMATION, NOT a bit-exact match (single-SSD lateral kernel +
convolution-at-iso + geometry conventions). Good enough as a PRIOR (beats naive vs GT; the DL
learns the residual), NOT a dose-of-record engine. Tighten via per-depth kernel broadening.
TODO: plan-level gamma A/B as the v12 prior channel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_KERNEL_NPZ = Path(__file__).resolve().parents[3] / "data/kernels/photons_Generic_kernel.npz"


def load_aaa_kernel(path=None) -> dict:
    """Load the exported pyRadPlan photon SVD kernel. Returns a dict with:
    betas (3,), m, penumbra, kernel_pos (Nr,), kernel_data (Nssd,3,Nr), kernel_ssds (Nssd,), SAD."""
    z = np.load(path or _KERNEL_NPZ)
    return {k: z[k] for k in z.files}


def depth_term(rd: torch.Tensor, betas, m: float) -> torch.Tensor:
    """Exact SVD depth build-up/falloff term, per component, evaluated at radiological
    depth `rd` (any shape). Returns (3, *rd.shape): β_c/(β_c−m)·(e^{−m·rd} − e^{−β_c·rd})."""
    b = torch.as_tensor(betas, dtype=torch.float32, device=rd.device).view(-1, *([1] * rd.ndim))
    rd = rd.unsqueeze(0)
    return b / (b - m) * (torch.exp(-m * rd) - torch.exp(-b * rd))


def radial_kernel_2d(kernel: dict, ssd: float, spacing_mm: float, half_mm: float = 60.0,
                     device="cpu") -> torch.Tensor:
    """Build the 3 SVD components as 2D lateral kernels K_c[x,z] on a grid of
    `spacing_mm`, radius up to `half_mm`, by interpolating each radial profile at the
    SSD nearest `ssd`. Returns (3, K, K) normalized for convolution (× spacing²)."""
    ssds = kernel["kernel_ssds"]; kpos = kernel["kernel_pos"]
    si = int(np.argmin(np.abs(ssds - ssd)))
    kdata = kernel["kernel_data"][si]                    # (3, Nr)
    r = int(round(half_mm / spacing_mm))
    ax = (torch.arange(-r, r + 1, device=device, dtype=torch.float32) * spacing_mm)
    xx, zz = torch.meshgrid(ax, ax, indexing="ij")
    rad = torch.sqrt(xx * xx + zz * zz)
    kp = torch.as_tensor(kpos, dtype=torch.float32, device=device)
    out = []
    for c in range(kdata.shape[0]):
        vals = torch.as_tensor(kdata[c], dtype=torch.float32, device=device)
        # linear interp of the radial profile at each |r|, 0 beyond the tabulated range
        idx = torch.clamp(torch.searchsorted(kp, rad.flatten()) - 1, 0, kp.numel() - 2)
        x0 = kp[idx]; x1 = kp[idx + 1]; y0 = vals[idx]; y1 = vals[idx + 1]
        w = ((rad.flatten() - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
        k = (y0 + w * (y1 - y0)).view_as(rad)
        k = torch.where(rad <= kp[-1], k, torch.zeros_like(k))
        out.append(k * spacing_mm * spacing_mm)          # conv normalization (1/mm² · mm²)
    return torch.stack(out, 0)


def aperture_bev(us: torch.Tensor, vs: torch.Tensor, machine, mlc_left, mlc_right,
                 device) -> torch.Tensor:
    """Binary open aperture A[v,u] on the BEV iso-plane grid, mirroring the per-voxel
    `open_mask` logic in physics/channels.py (MLC leaf pairs + jaws), but evaluated
    directly on the (vs, us) iso-plane grid. Returns (n_v, n_u) float {0,1}."""
    gv, gu = torch.meshgrid(vs, us, indexing="ij")            # (n_v, n_u) mm
    half = machine.num_leaf_pairs / 2.0
    pair = torch.floor(gv / machine.leaf_thickness_mm + half).long()
    valid = (pair >= 0) & (pair < machine.num_leaf_pairs)
    pidx = pair.clamp(0, machine.num_leaf_pairs - 1)
    ml = torch.as_tensor(mlc_left, dtype=torch.float32, device=device)[pidx]
    mr = torch.as_tensor(mlc_right, dtype=torch.float32, device=device)[pidx]
    jx, jy = machine.jaw_x_mm, machine.jaw_y_mm
    A = (valid & (ml < mr) & (gu >= ml) & (gu <= mr) &
         (gu >= jx[0]) & (gu <= jx[1]) &
         (gv >= jy[0]) & (gv <= jy[1])).float()
    return A


def aaa_prior_dose(density: np.ndarray, spacing, origin, source_xyz, axis, u_hat, v_hat,
                   iso_xyz, machine, mlc_left, mlc_right, kernel: dict, *,
                   bev_spacing_mm: float = 2.0, n_d: int = 256, ssd_mm: float | None = None,
                   pad_mm: float = 60.0, out_bbox: tuple | None = None,
                   coords: "torch.Tensor | None" = None,
                   device: str | None = None) -> np.ndarray:
    """SVD pencil-beam (AAA-class) dose for one control point, on the patient grid.

    Mirrors `raytrace.radiological_depth_fast`'s divergent BEV fan: it builds a
    (n_v, n_u, n_d) ray grid from the source through iso-plane points, accumulates
    radiological depth along each ray, and resamples back to every patient voxel.
    The lateral term is the OPEN APERTURE convolved (per SVD component) with the
    radial kernel in the iso plane; the depth term is `depth_term(rd)` per component;
    the result is divergence-scaled by (SAD / geo_depth)².

      dose(voxel) = [ Σ_c (A ⊛ K_c)(u,v) · depth_term_c(rd) ] · (SAD / geo_depth)²

    `rd` is radiological depth in MM water-equiv (our `rdepth` channel is g/cm² → ×10,
    which is the unit the kernel betas/m are calibrated in). `out_bbox`/`coords` follow
    the same fast-path contract as `radiological_depth_fast`. Absolute scale is arbitrary
    (kernel·fluence units) — LSQ-scale before comparing to GT, like the other priors.
    Returns a numpy array (dz,dy,dx) (bbox-shaped if `out_bbox` given, else full grid)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nz, ny, nx = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    dens = torch.as_tensor(density, dtype=torch.float32, device=dev)
    src = torch.as_tensor(source_xyz, dtype=torch.float32, device=dev)
    axis = torch.as_tensor(axis, dtype=torch.float32, device=dev)
    u_hat = torch.as_tensor(u_hat, dtype=torch.float32, device=dev)
    v_hat = torch.as_tensor(v_hat, dtype=torch.float32, device=dev)
    iso = torch.as_tensor(iso_xyz, dtype=torch.float32, device=dev)
    betas = kernel["betas"]
    m = float(kernel["m"])
    SAD = float(kernel["SAD"])

    # --- voxel -> iso-plane (u,v) and source distance (same projection as raytrace) ---
    if coords is None:
        xs = ox + torch.arange(nx, device=dev, dtype=torch.float32) * sx
        ys = oy + torch.arange(ny, device=dev, dtype=torch.float32) * sy
        zs = oz + torch.arange(nz, device=dev, dtype=torch.float32) * sz
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        P = torch.stack([gx, gy, gz], dim=-1)
    else:
        P = coords
    vec = P - src
    denom = (vec * axis).sum(-1)
    t = ((iso - src) * axis).sum() / torch.where(denom.abs() < 1e-6,
                                                 torch.full_like(denom, 1e-6), denom)
    hit = src + t.unsqueeze(-1) * vec
    rel = hit - iso
    vu = (rel * u_hat).sum(-1)
    vv = (rel * v_hat).sum(-1)
    vdist = torch.linalg.norm(vec, dim=-1)

    # --- BEV grid: EQUAL u/v spacing so the radial kernel convolution is isotropic.
    # extend the field extent by pad_mm so the convolution tails fit on the grid. ---
    u_max = float(vu.abs().max()) * 1.05 + pad_mm
    v_max = float(vv.abs().max()) * 1.05 + pad_mm
    n_u = max(int(round(2 * u_max / bev_spacing_mm)) + 1, 3)
    n_v = max(int(round(2 * v_max / bev_spacing_mm)) + 1, 3)
    us = torch.linspace(-u_max, u_max, n_u, device=dev)
    vs = torch.linspace(-v_max, v_max, n_v, device=dev)
    du = 2 * u_max / (n_u - 1)
    d_min = 1.0
    d_max = float(vdist.max()) * 1.02
    ds = torch.linspace(d_min, d_max, n_d, device=dev)
    step_cm = (d_max - d_min) / (n_d - 1) / 10.0

    # --- lateral: open aperture ⊛ radial kernel per SVD component, in the iso plane ---
    if ssd_mm is None:
        ssd_mm = SAD                                          # central-ray SSD proxy
    A = aperture_bev(us, vs, machine, mlc_left, mlc_right, dev)   # (n_v, n_u)
    K = radial_kernel_2d(kernel, ssd_mm, du, half_mm=pad_mm, device=dev)  # (3, Kk, Kk)
    pad = K.shape[-1] // 2
    fluence_c = F.conv2d(A.view(1, 1, n_v, n_u), K.unsqueeze(1),
                         padding=pad)[0]                      # (3, n_v, n_u)

    # --- BEV radiological depth along each ray (cumsum of density), then depth term ---
    gv, gu = torch.meshgrid(vs, us, indexing="ij")
    plane_pt = (iso.view(1, 1, 3) + gu.unsqueeze(-1) * u_hat.view(1, 1, 3)
                + gv.unsqueeze(-1) * v_hat.view(1, 1, 3))
    ray_dir = plane_pt - src.view(1, 1, 3)
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    bev_pts = src.view(1, 1, 1, 3) + ds.view(1, 1, n_d, 1) * ray_dir.unsqueeze(2)

    def world_to_norm(p):
        gx_ = (p[..., 0] - ox) / max(sx * (nx - 1), 1e-6) * 2 - 1
        gy_ = (p[..., 1] - oy) / max(sy * (ny - 1), 1e-6) * 2 - 1
        gz_ = (p[..., 2] - oz) / max(sz * (nz - 1), 1e-6) * 2 - 1
        return torch.stack([gx_, gy_, gz_], dim=-1)

    dens5 = dens.view(1, 1, nz, ny, nx)
    bev_norm = world_to_norm(bev_pts).view(1, n_v, n_u, n_d, 3)
    bev_dens = F.grid_sample(dens5, bev_norm, mode="bilinear", align_corners=True,
                             padding_mode="border").view(n_v, n_u, n_d)
    rd_mm = ((torch.cumsum(bev_dens, dim=-1) - 0.5 * bev_dens) * step_cm) * 10.0  # g/cm²→mm
    dterm = depth_term(rd_mm, betas, m)                       # (3, n_v, n_u, n_d)
    inv_sq = (SAD / ds.clamp_min(1.0)) ** 2                   # (n_d,)
    dose_bev = (fluence_c.unsqueeze(-1) * dterm).sum(0) * inv_sq   # (n_v, n_u, n_d)

    # --- resample BEV dose back to patient voxels (bbox-restricted fast-path) ---
    if out_bbox is not None:
        z0, z1, y0, y1, x0, x1 = out_bbox
        vu = vu[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        vv = vv[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        vdist = vdist[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    oz_, oy_, ox_ = vu.shape
    qn_u = vu / u_max
    qn_v = vv / v_max
    qn_d = (vdist - d_min) / max(d_max - d_min, 1e-6) * 2 - 1
    q = torch.stack([qn_d, qn_u, qn_v], dim=-1).view(1, oz_, oy_, ox_, 3)
    out = F.grid_sample(dose_bev.view(1, 1, n_v, n_u, n_d), q, mode="bilinear",
                        align_corners=True, padding_mode="border").view(oz_, oy_, ox_)
    return out.cpu().numpy()
