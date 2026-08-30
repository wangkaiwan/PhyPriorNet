"""End-to-end photon inference: image + beam plan + checkpoint -> per-CP dose
arrays on the full image grid. Reuses every existing module.

Runtime design (the challenge gates on avg wall-clock per dosemap):
  - per-IMAGE amortization: density (hu_to_density) and the full-volume world
    coordinate grid are built ONCE per image and reused across all its CPs (only
    gantry-dependent work — rdepth/projection — is recomputed per CP);
  - bbox fast-path: physics channels are computed only inside the beam bbox;
  - CP batching: CPs are streamed through the network in batches (padded to a
    common multiple-of-8 size) to saturate the GPU and amortize launch overhead.
"""
from __future__ import annotations

import numpy as np
import torch

from doserad.beam.photon_config import PhotonPlan
from doserad.data.dataset import DOSE_SCALE, _CH_SCALE, _NAIVE_SCALE
from doserad.io.mha import Volume
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.channels import photon_channels
from doserad.physics.density import hu_to_density
from doserad.physics.machine import PhotonMachine
from doserad.physics.naive_dose import SAD_MM, _SURFACE, _TAU


def _normalize_gpu(ch, add_naive):
    """GPU-resident equivalent of dataset.normalize_channels — keeps the channel
    stack on the GPU (avoids a per-CP GPU->CPU->GPU round-trip that cost ~200ms).
    Mirrors compute_naive_dose + _CH_SCALE exactly."""
    dev = ch.device
    if add_naive:
        rdepth, fluence, source_dist = ch[1], ch[2], ch[4]
        inv_sq = (SAD_MM / source_dist.clamp(min=1.0)) ** 2
        buildup = 1.0 - (1.0 - _SURFACE) * torch.exp(-rdepth / _TAU)
        naive = (fluence * inv_sq * buildup).unsqueeze(0)
        ch = torch.cat([ch, naive], dim=0)
        scale = torch.tensor(list(_CH_SCALE) + [_NAIVE_SCALE], dtype=torch.float32, device=dev)
    else:
        scale = torch.as_tensor(_CH_SCALE, dtype=torch.float32, device=dev)
    return ch / scale.view(-1, 1, 1, 1)


def _build_coords(image: Volume, dev: str) -> torch.Tensor:
    """Full-volume world (x,y,z) coordinate grid (z,y,x,3) — per-image constant."""
    nz, ny, nx = image.array.shape
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin
    xs = ox + torch.arange(nx, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(ny, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(nz, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)


def _flush_batch(net, buf, dev, full_shape, out, mod_idx, factor=8):
    """Run a batch of CP channel-crops through the net in one forward.

    Each crop has its own bbox size; pad all to a common multiple-of-`factor`
    size, stack, predict, then unpad each back to its bbox and scatter to the
    full grid keyed by (beam_idx, cp_idx)."""
    dims = [c.shape[-3:] for _, c, _ in buf]
    D = max(d[0] for d in dims); H = max(d[1] for d in dims); W = max(d[2] for d in dims)
    D = -(-D // factor) * factor; H = -(-H // factor) * factor; W = -(-W // factor) * factor
    B = len(buf); C = buf[0][1].shape[0]
    # crops are GPU tensors (C,d,h,w); assemble the batch on-device (no CPU round-trip)
    xb = torch.zeros((B, C, D, H, W), dtype=torch.float32, device=dev)
    for i, (_, c, _) in enumerate(buf):
        d, h, w = c.shape[-3:]
        xb[i, :, :d, :h, :w] = c.to(dev)
    m = torch.full((B,), mod_idx, dtype=torch.long, device=dev)
    with torch.no_grad():
        with torch.autocast(dev.split(":")[0], enabled=(dev != "cpu")):
            yb = net(xb, m).float()                          # (B,1,D,H,W)
    yb = (yb / DOSE_SCALE).cpu().numpy()                     # scaled-absolute -> Gy
    for i, (key, c, bbox) in enumerate(buf):
        d, h, w = c.shape[-3:]
        full = np.zeros(full_shape, dtype=np.float32)
        z0, z1, y0, y1, x0, x1 = bbox
        full[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = yb[i, 0, :d, :h, :w]
        out[key] = full


def photon_inference(image: Volume, plan: PhotonPlan, ckpt_path,
                     machine: PhotonMachine, *, modality: str = "ct",
                     pct_volume: np.ndarray | None = None,
                     device: str | None = None,
                     base_ch: int = 32, levels: int = 4,
                     in_ch: int = 5, add_naive: bool = False,
                     bottleneck: str = "plain",
                     bbox_margin: int = 8, infer_batch: int = 8,
                     max_batch_voxels: int = 2_500_000) -> dict:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = DoseUNet3D(in_ch=in_ch, base=base_ch, levels=levels, bottleneck=bottleneck).to(dev).eval()
    st = torch.load(ckpt_path, map_location=dev)
    net.load_state_dict(st.get("ema", st.get("model")))

    # --- per-image amortization: density + coordinate grid built ONCE ---
    if modality == "mri" and pct_volume is not None:
        density = hu_to_density(pct_volume, machine.hu_anchors)
    else:
        density = hu_to_density(image.array, machine.hu_anchors)
    # photon_channels builds its tensors on cuda-if-available regardless of the
    # net device, so coords must match that device (not necessarily `dev`).
    chan_dev = "cuda" if torch.cuda.is_available() else "cpu"
    coords = _build_coords(image, chan_dev)
    mod_idx = 1 if modality == "mri" else 0
    full_shape = image.array.shape

    out: dict = {}
    buf: list = []
    mx = [0, 0, 0]                 # running per-dim max of the current batch
    for beam in plan.beams:
        for cp in beam.control_points:
            crop, bbox = photon_channels(
                image=image, machine=machine, iso_xyz=beam.iso_center,
                gantry_deg=cp.gantry_angle,
                mlc_left=np.asarray(cp.mlc_left_int_mm),
                mlc_right=np.asarray(cp.mlc_right_int_mm),
                density_override=density, coords=coords,
                crop_margin=bbox_margin, return_tensor=True)   # stay on GPU
            crop = _normalize_gpu(crop, add_naive)
            d, h, w = crop.shape[-3:]
            nmx = [max(mx[0], d), max(mx[1], h), max(mx[2], w)]
            # flush before adding if this crop would blow the batch's padded-voxel
            # budget or the count cap (memory-safe regardless of crop sizes)
            if buf and ((len(buf) + 1) * nmx[0] * nmx[1] * nmx[2] > max_batch_voxels
                        or len(buf) >= infer_batch):
                _flush_batch(net, buf, dev, full_shape, out, mod_idx)
                buf = []; mx = [0, 0, 0]; nmx = [d, h, w]
            buf.append(((int(beam.beam_idx), int(cp.cp_idx)), crop, bbox))
            mx = nmx
    if buf:
        _flush_batch(net, buf, dev, full_shape, out, mod_idx)
    return out
