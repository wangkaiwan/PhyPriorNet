"""Fast GPU (PyTorch) port of the pyRadPlan "Generic" proton Hong analytical pencil-beam
dose (single-Gaussian lateral model) — the `pb_prior` channel of the with-prior proton model.

NEW file. Reuses the photon ray-trace (radiological_depth_fast) for WEPL only; does NOT
modify any committed CT/photon code. Self-contained at inference: machine base data is read
from a precomputed npz (`data/proton_machine_generic.npz`, exported from pyRadPlan once), so
the `pyradplan` conda env is NOT required at inference time.

ALGORITHM (mirrors pyRadPlan ParticleHongPencilBeamEngine, lateral_model="single", which is
what auto-selection picks for the Generic machine, i.e. the prior the model was trained on):

    dose(voxel) = comp_fac * IDD(d) / (2*pi*sigma^2) * exp(-r^2 / (2*sigma^2))

  with
    d        = rad_depth (WEPL, g/cm^2 ~ mm in water) at the voxel, minus rad_depth_offset
    IDD(d)   = conversion_factor * machine integrated-depth-dose, interpolated at d in mm
               (conversion_factor = 1.6021766208e-2 MeV cm^2/g/primary -> Gy mm^2 / 1e6 prim)
    sigma^2  = kernel.sigma(d)^2 + sigma_ini^2        (both mm)
    sigma_ini= initial spot size at the surface, interp of focus.sigma over focus.dist at SSD
    r        = lateral distance (mm) from the voxel to the pencil central axis
    rad_depth_offset = 0.0011*(SSD + bams_to_iso_dist - SAD - fit_air_offset)  (air offset corr)
    comp_fac = lateral-cutoff compensation factor (~1/cut_off_level); using the default 1.0,
               which matches a 100% lateral cutoff (geometric_lateral_cutoff only). pyRadPlan's
               dosimetric cutoff yields ~1.005; we expose `comp_fac` to allow matching.

GEOMETRY: pyRadPlan overrides the JSON SAD with the machine SAD (1e4 mm) and places the source
at iso_center(=ray_target) + R@[0,-SAD,0]. For gantry where ray_source->ray_target is collinear
with that, WEPL is identical; we replicate the machine-SAD source so divergence matches too.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import torch

from doserad.physics.density import hu_to_density

_DEFAULT_NPZ = Path(__file__).resolve().parents[2] / "data" / "proton_machine_generic.npz"
_SSD_DENSITY_THRESHOLD = 0.05   # pyRadPlan ssd_density_threshold (skin)


class ProtonMachineData:
    """Holds the exported pyRadPlan Generic base data as torch tensors (on `device`)."""

    def __init__(self, npz_path=_DEFAULT_NPZ, device="cpu"):
        d = np.load(npz_path)
        self.device = device
        self.energies = torch.as_tensor(d["energies"], dtype=torch.float32, device=device)
        self.lengths = torch.as_tensor(d["lengths"], dtype=torch.long, device=device)
        # padded (N, maxlen) with NaN beyond `lengths`
        self.depths = torch.as_tensor(np.nan_to_num(d["depths"], nan=0.0), dtype=torch.float32, device=device)
        self.idd = torch.as_tensor(np.nan_to_num(d["idd"], nan=0.0), dtype=torch.float32, device=device)
        self.sigma = torch.as_tensor(np.nan_to_num(d["sigma"], nan=0.0), dtype=torch.float32, device=device)
        # double-Gaussian (nuclear-halo) lateral params — used only when DOSERAD_LATERAL_DOUBLE=1.
        # sigma1=narrow core, sigma2=wide halo (~9x), weight=fraction in the halo (~0.03-0.34).
        self.sigma1 = torch.as_tensor(np.nan_to_num(d["sigma1"], nan=0.0), dtype=torch.float32, device=device) if "sigma1" in d.files else None
        self.sigma2 = torch.as_tensor(np.nan_to_num(d["sigma2"], nan=0.0), dtype=torch.float32, device=device) if "sigma2" in d.files else None
        self.weight = torch.as_tensor(np.nan_to_num(d["weight"], nan=0.0), dtype=torch.float32, device=device) if "weight" in d.files else None
        self.offset = torch.as_tensor(d["offset"], dtype=torch.float32, device=device)
        self.foc_dist = torch.as_tensor(d["foc_dist"], dtype=torch.float32, device=device)
        self.foc_sigma = torch.as_tensor(d["foc_sigma"], dtype=torch.float32, device=device)
        self.sad = float(d["sad"])
        self.bams_to_iso_dist = float(d["bams_to_iso_dist"])
        self.fit_air_offset = float(d["fit_air_offset"])
        self.conversion_factor = float(d["conversion_factor"])
        self._energies_np = d["energies"].astype(np.float64)

    def energy_index(self, energy: float) -> int:
        """Nearest machine energy (matches pyRadPlan nearest-neighbour snapping)."""
        return int(np.argmin(np.abs(self._energies_np - float(energy))))

    def sigma_ini(self, eidx: int, ssd: float) -> float:
        """Initial spot sigma (mm) at the surface, linear interp of focus over distance."""
        dist = self.foc_dist.cpu().numpy()
        sig = self.foc_sigma[eidx].cpu().numpy()
        return float(np.interp(ssd, dist, sig))   # numpy extrapolates by clamping; ssd is in-range


def _wepl_crop(density, spacing, origin, src, coords, dev, step_mm: float = 1.0,
               march_start_mm: float | None = None, chunk: int = 8_000_000):
    """Accurate radiological depth (WEPL, g/cm^2) source->voxel for every voxel in `coords`
    (..., 3 world-xyz). Straight-line march from `src`, trilinear density sampling, sum*step(cm).

    The BEV-fan `radiological_depth_fast` over-counts here (tilted gantry + anisotropic z),
    biasing the Bragg peak ~30% deep; this direct per-voxel march matches the exact integral.

    `march_start_mm`: skip integrating the (near-vacuum) air gap before the patient. The proton
    source is machine-SAD (~1e4 mm) away but the patient spans only the last few hundred mm, so
    we start the integral at `march_start_mm` from the source (default: just before the closest
    crop voxel) — air (rho~1e-3) contributes negligibly and this cuts steps ~30x.
    Distance-to-voxel varies, so per-step we mask out samples beyond each voxel's own distance.
    """
    nz, ny, nx = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    dens5 = (density.to(dev, torch.float32) if torch.is_tensor(density)
             else torch.as_tensor(density, dtype=torch.float32, device=dev)).view(1, 1, nz, ny, nx)
    src_t = torch.as_tensor(src, dtype=torch.float32, device=dev)

    shp = coords.shape[:-1]
    P = coords.reshape(-1, 3)
    vec = P - src_t
    dist = torch.linalg.norm(vec, dim=-1)                              # mm to each voxel
    direction = vec / dist.clamp_min(1e-6).unsqueeze(-1)

    if march_start_mm is None:
        # start ~50mm before the nearest crop voxel (patient surface is upstream of the crop too,
        # but the crop is GT-derived and starts at/inside the patient; back off generously)
        march_start_mm = max(float(dist.min().item()) - 200.0, 0.0)
    n_steps = int(torch.ceil((dist.max() - march_start_mm) / step_mm).item()) + 1
    if n_steps < 1:   # degenerate: march_start past the farthest voxel (e.g. SSD detection fell back to
                      # SAD on a patch wholly upstream of the detected skin) -> fall back to a local march
        march_start_mm = max(float(dist.min().item()) - 50.0, 0.0)
        n_steps = max(int(torch.ceil((dist.max() - march_start_mm) / step_mm).item()) + 1, 1)

    out = torch.zeros(P.shape[0], dtype=torch.float32, device=dev)
    inv_w = 2.0 / max(sx * (nx - 1), 1e-6)
    inv_h = 2.0 / max(sy * (ny - 1), 1e-6)
    inv_d = 2.0 / max(sz * (nz - 1), 1e-6)
    csz = max(1, chunk // max(n_steps, 1))
    t = march_start_mm + (torch.arange(n_steps, device=dev, dtype=torch.float32) + 0.5) * step_mm
    for s0 in range(0, P.shape[0], csz):
        s1 = min(s0 + csz, P.shape[0])
        d_c = direction[s0:s1]; dist_c = dist[s0:s1]                   # (m,3),(m,)
        m = d_c.shape[0]
        pts = src_t.view(1, 1, 3) + t.view(1, -1, 1) * d_c.view(m, 1, 3)              # (m,S,3)
        gx = (pts[..., 0] - ox) * inv_w - 1
        gy = (pts[..., 1] - oy) * inv_h - 1
        gz = (pts[..., 2] - oz) * inv_d - 1
        grid = torch.stack([gx, gy, gz], dim=-1).view(1, 1, m, n_steps, 3)
        sampled = torch.nn.functional.grid_sample(
            dens5, grid, mode="bilinear", align_corners=True,
            padding_mode="zeros").view(m, n_steps)
        active = (t.view(1, -1) < dist_c.view(-1, 1)).float()
        out[s0:s1] = (sampled * active).sum(dim=1) * (step_mm / 10.0)  # mm->cm => g/cm^2
    return out.reshape(shp)


def _interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of fp(xp) at points x (all 1-D, xp ascending). Clamps at ends."""
    idx = torch.searchsorted(xp, x.clamp(xp[0], xp[-1]))
    idx = idx.clamp(1, xp.numel() - 1)
    x0 = xp[idx - 1]; x1 = xp[idx]
    y0 = fp[idx - 1]; y1 = fp[idx]
    t = (x - x0) / (x1 - x0).clamp_min(1e-9)
    return y0 + t * (y1 - y0)


def proton_pb_dose_gpu(image, ray_source, ray_target, energy, *, out_bbox,
                       machine: ProtonMachineData, hu_anchors=None, density_override=None,
                       comp_fac: float = 1.0, device: str | None = None,
                       use_machine_sad_source: bool = True,
                       mask_air: bool = False, air_thr: float = 0.1, body_mask=None,
                       return_tensor: bool = False):
    """Compute the proton Hong PB dose on the crop `out_bbox`=(z0,z1,y0,y1,x0,x1) (inclusive).

    `image`   : Volume (array (z,y,x), spacing (sx,sy,sz), origin (ox,oy,oz)) — x-first metadata.
    `ray_source`,`ray_target` : world-xyz (mm), from the plan JSON.
    Returns (dose_np (dz,dy,dx) Gy-ish) — or a gpu tensor if return_tensor.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nz, ny, nx = image.array.shape
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin

    if density_override is not None:
        if torch.is_tensor(density_override):
            density = density_override.to(dev, torch.float32)   # differentiable path: keep autograd graph
        else:
            density = np.asarray(density_override, dtype=np.float32)
    else:
        density = hu_to_density(image.array, hu_anchors).astype(np.float32)
    dens_is_tensor = torch.is_tensor(density)

    tgt = np.asarray(ray_target, dtype=np.float64)
    jsrc = np.asarray(ray_source, dtype=np.float64)
    axis = tgt - jsrc
    axis = (axis / (np.linalg.norm(axis) + 1e-12))

    # pyRadPlan places the source machine-SAD upstream of the iso (= ray_target) along the ray.
    if use_machine_sad_source:
        src = (tgt - axis * machine.sad).astype(np.float32)
    else:
        src = jsrc.astype(np.float32)
    axis_f = axis.astype(np.float32)

    z0, z1, y0, y1, x0, x1 = out_bbox

    # --- crop world coords (shared by WEPL + lateral distance) ---
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)                          # (dz,dy,dx,3)

    # --- SSD along central ray: distance source -> first voxel with density > threshold ---
    ssd = _compute_ssd(density.detach() if dens_is_tensor else density, image.spacing, image.origin,
                       src, axis_f, machine.sad, dev, threshold=_SSD_DENSITY_THRESHOLD)

    # --- WEPL (g/cm^2) on the crop, accurate per-voxel march; skip the air gap before the skin ---
    wepl = _wepl_crop(density, image.spacing, image.origin, src, coords, dev,
                      march_start_mm=max(ssd - 50.0, 0.0))            # (dz,dy,dx)

    # --- lateral distance to the pencil axis ---
    src_t = torch.as_tensor(src, device=dev)
    axis_t = torch.as_tensor(axis_f, device=dev)
    rel = coords - src_t
    along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
    lateral_dist = torch.linalg.norm(rel - along, dim=-1)              # mm (dz,dy,dx)

    eidx = machine.energy_index(energy)
    n = int(machine.lengths[eidx].item())
    depths = machine.depths[eidx, :n]                                   # mm
    idd = machine.conversion_factor * machine.idd[eidx, :n]
    sigma_d = machine.sigma[eidx, :n]                                   # mm
    offset = float(machine.offset[eidx].item())

    sigma_ini = machine.sigma_ini(eidx, ssd)
    sigma_ini_sq = sigma_ini ** 2
    rad_depth_offset = 0.0011 * (ssd + machine.bams_to_iso_dist - machine.sad - machine.fit_air_offset)

    # WEPL is in g/cm^2 == mm of water (rho_water=1 g/cm^3 -> 1 g/cm^2 = 10mm? no: 1 g/cm^2 = 1cm water = 10mm)
    # radiological_depth_fast returns density(g/cm^3) * path(cm) = g/cm^2; depth in water mm = value*10.
    rad_depths_mm = wepl * 10.0                                        # g/cm^2 -> mm water

    # depth coordinate into the kernel tables: kernel.depths + offset - rad_depth_offset
    eff_depths = depths + offset - rad_depth_offset
    d = rad_depths_mm.reshape(-1)

    idd_v = _interp_1d(d, eff_depths, idd)
    sig_v = _interp_1d(d, eff_depths, sigma_d)
    # zero dose beyond the kernel range (pyRadPlan masks voxels past depths[-1])
    out_of_range = (d > eff_depths[-1]) | (d < eff_depths[0])
    idd_v = torch.where(out_of_range, torch.zeros_like(idd_v), idd_v)

    r2 = (lateral_dist.reshape(-1)) ** 2
    if os.environ.get("DOSERAD_LATERAL_DOUBLE") == "1" and machine.sigma1 is not None:
        # Hong DOUBLE-Gaussian lateral (nuclear-interaction halo): narrow core sigma1 + wide halo sigma2,
        # weight = fraction in the wide component. pyRadPlan lateral_model="double". Adds the ~10% low-dose
        # lateral halo that the single effective `sigma` fit drops. OPT-IN; default single path unchanged.
        sig1_v = _interp_1d(d, eff_depths, machine.sigma1[eidx, :n])
        sig2_v = _interp_1d(d, eff_depths, machine.sigma2[eidx, :n])
        w_v = _interp_1d(d, eff_depths, machine.weight[eidx, :n]).clamp(0.0, 1.0)
        s1sq = sig1_v ** 2 + sigma_ini_sq
        s2sq = sig2_v ** 2 + sigma_ini_sq
        g1 = torch.exp(-r2 / (2.0 * s1sq)) / (2.0 * torch.pi * s1sq)
        g2 = torch.exp(-r2 / (2.0 * s2sq)) / (2.0 * torch.pi * s2sq)
        lateral = (1.0 - w_v) * g1 + w_v * g2
    else:
        sigma_sq = sig_v ** 2 + sigma_ini_sq
        lateral = torch.exp(-r2 / (2.0 * sigma_sq)) / (2.0 * torch.pi * sigma_sq)
    dose = comp_fac * lateral * idd_v
    dose = dose.reshape(rad_depths_mm.shape).clamp_min(0.0)

    if mask_air:   # zero dose in EXTERNAL air (MC deposits ~0 there; removes IDD(0) painted on the
                   # upstream air gap where WEPL~0). With body_mask, only outside-body voxels are zeroed
                   # (internal gas cavities kept); else fall back to a plain density threshold.
        if body_mask is not None:
            bm = torch.as_tensor(body_mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1], device=dev).to(dose.dtype)
        else:
            dc = torch.as_tensor(density[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1], dtype=torch.float32, device=dev)
            bm = (dc >= air_thr).to(dose.dtype)
        dose = dose * bm

    if return_tensor:
        return dose
    return dose.cpu().numpy().astype(np.float32)


def _compute_ssd(density, spacing, origin, src, axis, sad, dev, threshold=0.05, step_mm=1.0):
    """Source-to-surface distance: march along the central ray from the source, return the
    distance to the first sample with density > threshold (matches pyRadPlan skin SSD)."""
    nz, ny, nx = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    dens = torch.as_tensor(density, dtype=torch.float32, device=dev).view(1, 1, nz, ny, nx)
    src_t = torch.as_tensor(src, dtype=torch.float32, device=dev)
    axis_t = torch.as_tensor(axis, dtype=torch.float32, device=dev)

    # march only within a plausible window: from sad-300 to sad+300 mm (patient near iso)
    t = torch.arange(sad - 350.0, sad + 50.0, step_mm, device=dev)
    pts = src_t.view(1, 3) + t.view(-1, 1) * axis_t.view(1, 3)          # (M,3)
    gx_ = (pts[:, 0] - ox) / max(sx * (nx - 1), 1e-6) * 2 - 1
    gy_ = (pts[:, 1] - oy) / max(sy * (ny - 1), 1e-6) * 2 - 1
    gz_ = (pts[:, 2] - oz) / max(sz * (nz - 1), 1e-6) * 2 - 1
    grid = torch.stack([gx_, gy_, gz_], dim=-1).view(1, 1, 1, -1, 3)
    sampled = torch.nn.functional.grid_sample(dens, grid, mode="bilinear",
                                              align_corners=True, padding_mode="zeros").view(-1)
    hit = (sampled > threshold).nonzero()
    if hit.numel() == 0:
        return float(sad)
    return float(t[int(hit[0].item())].item())
