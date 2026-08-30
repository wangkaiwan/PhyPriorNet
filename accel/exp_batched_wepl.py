"""Batched proton engine — milestone ①: WEPL reformulation bench (2026-08-26).

Two candidate accelerations of the deploy WEPL (the FLOP floor of the proton build), benched on one
patient's real beam geometry vs the deployed per-voxel march (_wepl_crop == wepl_fused source-ref):

  V1  launch-amortized batch: all beamlets' voxels through wepl_fused in giant chunks with a
      per-voxel src tensor. EXACT by construction; expected gain small on 5090 (FLOP-bound), larger
      on launch-heavy platforms.
  V2  BEV-cumsum: per beam, resample density onto a source-centred beam-aligned grid (1 grid_sample),
      cumsum along depth (= the whole WEPL in one op), trilinear-sample WEPL back at voxel positions.
      ~O(grid) instead of O(voxels x steps): potential 10-100x. NOT exact — the number that matters
      is max|Δ| vs the deployed march (train/deploy channel consistency needs <~2e-3 in scaled ch1;
      WEPL is in g/cm^2, ch scale /_P_CH_SCALE ~ /30 → raw WEPL tolerance ~0.06 g/cm^2).

Usage: CUDA_VISIBLE_DEVICES=1 python accel/exp_batched_wepl.py [pid] [n_rays]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from accel.wepl_fused import wepl_fused
from container.proton.geom_bbox import geom_bbox_proton

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
NRAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DEV = "cuda"


def coords_of_bbox(bb, spacing, origin, dev):
    z0, z1, y0, y1, x0, x1 = bb
    sx, sy, sz = spacing; ox, oy, oz = origin
    zz = torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz + oz
    yy = torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy + oy
    xx = torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx + ox
    gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)


def wepl_bev_cumsum(dens5, spacing, origin, src, coords, step_mm=1.0, pad_mm=30.0):
    """V2: BEV grid WEPL. Build a beam-aligned grid spanning the crop's solid angle from src,
    resample density on it, cumsum along depth, sample back at coords. All ops O(grid)."""
    nz, ny, nx = dens5.shape[-3:]
    sx, sy, sz = spacing; ox, oy, oz = origin
    P = coords.reshape(-1, 3)
    src_t = torch.as_tensor(src, dtype=torch.float32, device=P.device)
    vec = P - src_t
    dist = torch.linalg.norm(vec, dim=-1)
    dmax = float(dist.max())
    # beam axis = mean direction; u,v span
    axis = F.normalize(vec.mean(0), dim=0)
    tmp = torch.tensor([1.0, 0, 0], device=P.device)
    if abs(float(torch.dot(tmp, axis))) > 0.9:
        tmp = torch.tensor([0, 1.0, 0], device=P.device)
    u = F.normalize(torch.linalg.cross(axis, tmp), dim=0)
    v = torch.linalg.cross(axis, u)
    # angular extent of the crop as seen from src (+pad)
    du = (vec @ u) / (vec @ axis).clamp_min(1e-3)   # tan angles
    dv = (vec @ v) / (vec @ axis).clamp_min(1e-3)
    du0, du1 = float(du.min()), float(du.max())
    dv0, dv1 = float(dv.min()), float(dv.max())
    pad_t = pad_mm / dmax
    du0 -= pad_t; du1 += pad_t; dv0 -= pad_t; dv1 += pad_t
    # BEV grid resolution: ~step_mm at depth dmax laterally, step_mm in depth
    nu = max(int((du1 - du0) * dmax / step_mm) + 2, 8)
    nv = max(int((dv1 - dv0) * dmax / step_mm) + 2, 8)
    nd = int(dmax / step_mm) + 2
    tu = torch.linspace(du0, du1, nu, device=P.device)
    tv = torch.linspace(dv0, dv1, nv, device=P.device)
    td = (torch.arange(nd, device=P.device, dtype=torch.float32) + 0.5) * step_mm
    # world position of each BEV node: src + d*(axis + tu*u + tv*v)/norm  (fan geometry, exact rays)
    dirs = axis.view(1, 1, 3) + tu.view(-1, 1, 1) * u.view(1, 1, 3) + tv.view(1, -1, 1) * v.view(1, 1, 3)
    dirs = F.normalize(dirs, dim=-1)                       # (nu,nv,3)
    pts = src_t.view(1, 1, 1, 3) + td.view(-1, 1, 1, 1) * dirs.view(1, nu, nv, 3)   # (nd,nu,nv,3)
    inv_w = 2.0 / max(sx * (nx - 1), 1e-6)
    inv_h = 2.0 / max(sy * (ny - 1), 1e-6)
    inv_d = 2.0 / max(sz * (nz - 1), 1e-6)
    g = torch.stack([(pts[..., 0] - ox) * inv_w - 1,
                     (pts[..., 1] - oy) * inv_h - 1,
                     (pts[..., 2] - oz) * inv_d - 1], dim=-1)[None]     # (1,nd,nu,nv,3)
    rho = F.grid_sample(dens5, g, mode="bilinear", align_corners=True,
                        padding_mode="zeros")[0, 0]                      # (nd,nu,nv)
    wepl_grid = torch.cumsum(rho, dim=0) * (step_mm / 10.0)              # source-ref WEPL at node centers
    # sample back: voxel -> (depth d=|vec|, angle tu,tv) -> normalized coords in the BEV grid
    qd = (dist / step_mm - 0.5) / max(nd - 1, 1) * 2 - 1
    qu = (du - du0) / max(du1 - du0, 1e-9) * 2 - 1
    qv = (dv - dv0) / max(dv1 - dv0, 1e-9) * 2 - 1
    q = torch.stack([qv, qu, qd], dim=-1).view(1, -1, 1, 1, 3)           # grid_sample xyz = (v,u,d)
    out = F.grid_sample(wepl_grid[None, None], q, mode="bilinear",
                        align_corners=True, padding_mode="border").view(-1)
    return out.reshape(coords.shape[:-1])


def main():
    pdir = Path(ROOT) / PID
    ct = load_mha(pdir / "image" / "ct.mha")
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    pm = ProtonMachineData(device=DEV)
    dens = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
    dens_t = torch.as_tensor(dens, device=DEV)
    dens5 = dens_t.view(1, 1, *dens.shape)
    plan = json.load(open(pdir / f"{PID}.json"))
    rays = [(r["ray_source"], r["ray_target"], bl["energy"])
            for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]][: NRAYS]
    print(f"[bench] {PID}: {len(rays)} beamlets", flush=True)

    # collect per-beamlet geom bboxes + coords
    items = []
    for src, tgt, e in rays:
        gb = geom_bbox_proton(dens, ct.spacing, ct.origin, src, tgt, pm, e)
        if gb is None: continue
        items.append((src, coords_of_bbox(gb, ct.spacing, ct.origin, DEV)))
    nvox = sum(c.reshape(-1, 3).shape[0] for _, c in items)
    print(f"[bench] {len(items)} bboxes, {nvox/1e6:.1f}M voxels total", flush=True)

    # ---- reference: per-beamlet wepl_fused (deploy path) ----
    torch.cuda.synchronize(); t0 = time.time()
    ref = [wepl_fused(dens_t, ct.spacing, ct.origin, src, c, DEV)[0] for src, c in items]
    torch.cuda.synchronize(); t_ref = time.time() - t0
    print(f"V0 per-beamlet march : {t_ref*1000:.0f} ms ({t_ref*1000/len(items):.1f} ms/beamlet)", flush=True)

    # ---- V2: BEV cumsum per beamlet ----
    torch.cuda.synchronize(); t0 = time.time()
    bev = [wepl_bev_cumsum(dens5, ct.spacing, ct.origin, src, c) for src, c in items]
    torch.cuda.synchronize(); t_bev = time.time() - t0
    dmax = max(float((b - r).abs().max()) for b, r in zip(bev, ref))
    dmean = float(np.mean([float((b - r).abs().mean()) for b, r in zip(bev, ref)]))
    print(f"V2 BEV-cumsum        : {t_bev*1000:.0f} ms ({t_bev*1000/len(items):.1f} ms/beamlet)  "
          f"speedup {t_ref/t_bev:.1f}x", flush=True)
    print(f"V2 accuracy vs march : max|Δ| {dmax:.4f} g/cm2, mean|Δ| {dmean:.5f}  "
          f"(need max ≲ 0.06 for channel-scale 2e-3)", flush=True)




def wepl_beam_ortho(dens5, spacing, origin, beam_dir, items_of_beam, lat_mm=1.0, step_mm=1.0,
                    skin_thr=0.05):
    """V3: ONE orthographic BEV grid per beam (rays exactly parallel -> each BEV column IS a true
    ray). Covers the union of the beam's crops laterally; depth from an upstream reference plane
    (air above skin contributes ~0, matching the from-source march). Returns per-item
    (w_src, w_skin, entered) sampled at the item's coords. items_of_beam: list of coords tensors."""
    nz, ny, nx = dens5.shape[-3:]
    sx, sy, sz = spacing; ox, oy, oz = origin
    dev = items_of_beam[0].device
    d = F.normalize(torch.as_tensor(beam_dir, dtype=torch.float32, device=dev), dim=0)
    tmp = torch.tensor([1.0, 0, 0], device=dev)
    if abs(float(torch.dot(tmp, d))) > 0.9:
        tmp = torch.tensor([0, 1.0, 0], device=dev)
    u = F.normalize(torch.linalg.cross(d, tmp), dim=0)
    v = torch.linalg.cross(d, u)
    allP = torch.cat([c.reshape(-1, 3) for c in items_of_beam])
    pu = allP @ u; pv = allP @ v; pd = allP @ d
    # depth reference plane: upstream of every crop voxel minus a pad; lateral extent + pad
    pad = 8.0
    u0, u1 = float(pu.min()) - pad, float(pu.max()) + pad
    v0, v1 = float(pv.min()) - pad, float(pv.max()) + pad
    d0, d1 = float(pd.min()) - 250.0, float(pd.max()) + pad   # 250mm upstream buffer (oblique-entry safe)
    nu = int((u1 - u0) / lat_mm) + 2
    nv = int((v1 - v0) / lat_mm) + 2
    nd = int((d1 - d0) / step_mm) + 2
    tu = torch.linspace(u0, u1, nu, device=dev)
    tv = torch.linspace(v0, v1, nv, device=dev)
    td = d0 + (torch.arange(nd, device=dev, dtype=torch.float32) + 0.5) * step_mm
    # world coords of BEV nodes: p = td*d + tu*u + tv*v   (orthographic)
    pts = (td.view(-1, 1, 1, 1) * d.view(1, 1, 1, 3) + tu.view(1, -1, 1, 1) * u.view(1, 1, 1, 3)
           + tv.view(1, 1, -1, 1) * v.view(1, 1, 1, 3))
    inv_w = 2.0 / max(sx * (nx - 1), 1e-6); inv_h = 2.0 / max(sy * (ny - 1), 1e-6); inv_d = 2.0 / max(sz * (nz - 1), 1e-6)
    g = torch.stack([(pts[..., 0] - ox) * inv_w - 1, (pts[..., 1] - oy) * inv_h - 1,
                     (pts[..., 2] - oz) * inv_d - 1], dim=-1)[None]
    rho = F.grid_sample(dens5, g, mode="bilinear", align_corners=True, padding_mode="zeros")[0, 0]
    w_src_g = torch.cumsum(rho, dim=0) * (step_mm / 10.0)
    hit = (rho > skin_thr).float()
    crossed = (torch.cumsum(hit, dim=0) > 0).float()
    w_skin_g = torch.cumsum(rho * crossed, dim=0) * (step_mm / 10.0)
    ent_g = crossed          # entered = crossed-skin-by-this-depth
    outs = []
    for c in items_of_beam:
        P = c.reshape(-1, 3)
        qu = ((P @ u) - u0) / max(u1 - u0, 1e-9) * 2 - 1
        qv = ((P @ v) - v0) / max(v1 - v0, 1e-9) * 2 - 1
        qd = (((P @ d) - d0) / step_mm - 0.5) / max(nd - 1, 1) * 2 - 1
        q = torch.stack([qv, qu, qd], dim=-1).view(1, -1, 1, 1, 3)
        ws = F.grid_sample(w_src_g[None, None], q, mode="bilinear", align_corners=True,
                           padding_mode="border").view(-1).reshape(c.shape[:-1])
        wk = F.grid_sample(w_skin_g[None, None], q, mode="bilinear", align_corners=True,
                           padding_mode="border").view(-1).reshape(c.shape[:-1])
        en = F.grid_sample(ent_g[None, None], q, mode="bilinear", align_corners=True,
                           padding_mode="border").view(-1).reshape(c.shape[:-1])
        outs.append((ws, wk, en))
    return outs


def main_v3():
    pdir = Path(ROOT) / PID
    ct = load_mha(pdir / "image" / "ct.mha")
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    pm = ProtonMachineData(device=DEV)
    dens = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
    dens_t = torch.as_tensor(dens, device=DEV)
    dens5 = dens_t.view(1, 1, *dens.shape)
    plan = json.load(open(pdir / f"{PID}.json"))
    b = plan["beams"][0]
    import numpy as _np
    bd = _np.array(b["rays"][0]["ray_target"]) - _np.array(b["rays"][0]["ray_source"])
    items, srcs = [], []
    for r in b["rays"]:
        for bl in r["beamlets"]:
            gb = geom_bbox_proton(dens, ct.spacing, ct.origin, r["ray_source"], r["ray_target"], pm, bl["energy"])
            if gb is None: continue
            items.append(coords_of_bbox(gb, ct.spacing, ct.origin, DEV)); srcs.append(r["ray_source"])
    print(f"[V3] beam0: {len(items)} beamlets, {sum(c.reshape(-1,3).shape[0] for c in items)/1e6:.1f}M vox", flush=True)
    torch.cuda.synchronize(); t0 = time.time()
    ref = [wepl_fused(dens_t, ct.spacing, ct.origin, s, c, DEV) for s, c in zip(srcs, items)]
    torch.cuda.synchronize(); t_ref = time.time() - t0
    torch.cuda.synchronize(); t0 = time.time()
    v3 = wepl_beam_ortho(dens5, ct.spacing, ct.origin, bd, items)
    torch.cuda.synchronize(); t_v3 = time.time() - t0
    for name, idx in (("w_src", 0), ("w_skin", 1)):
        dmax = max(float((a[idx] - r[idx]).abs().max()) for a, r in zip(v3, ref))
        dmean = float(np.mean([float((a[idx] - r[idx]).abs().mean()) for a, r in zip(v3, ref)]))
        print(f"[V3] {name}: max|Δ| {dmax:.4f}  mean|Δ| {dmean:.5f}", flush=True)
    print(f"[V3] per-beamlet march {t_ref*1000/len(items):.1f} ms/bl  ->  beam-ortho {t_v3*1000/len(items):.2f} ms/bl  "
          f"speedup {t_ref/t_v3:.1f}x", flush=True)




if __name__ == "__main__":
    (main_v3() if (len(sys.argv) > 3 and sys.argv[3] == "v3") else main())


def wepl_beam_grid(dens5, spacing, origin, beam_dir, u0u1v0v1d0d1=None, coords_probe=None,
                   lat_mm=1.0, step_mm=1.0, skin_thr=0.05):
    """Memory-lean per-beam BEV grids: returns (basis, extents, w_src_g, w_skin_g, ent_g) with the
    cumsum grids in FP16 and all intermediates freed. Extents from coords_probe (any coords tensor
    covering the beam's lateral/depth span, e.g. concat of geom-box corner points)."""
    import torch as T, torch.nn.functional as FF
    nz, ny, nx = dens5.shape[-3:]
    sx, sy, sz = spacing; ox, oy, oz = origin
    dev = dens5.device
    d = FF.normalize(T.as_tensor(beam_dir, dtype=T.float32, device=dev), dim=0)
    tmp = T.tensor([1.0, 0, 0], device=dev)
    if abs(float(T.dot(tmp, d))) > 0.9:
        tmp = T.tensor([0, 1.0, 0], device=dev)
    u = FF.normalize(T.linalg.cross(d, tmp), dim=0)
    v = T.linalg.cross(d, u)
    P = coords_probe.reshape(-1, 3)
    pu = P @ u; pv = P @ v; pd = P @ d
    pad = 8.0
    u0, u1 = float(pu.min()) - pad, float(pu.max()) + pad
    v0, v1 = float(pv.min()) - pad, float(pv.max()) + pad
    d0, d1 = float(pd.min()) - 250.0, float(pd.max()) + pad
    del P, pu, pv, pd
    nu = int((u1 - u0) / lat_mm) + 2; nv = int((v1 - v0) / lat_mm) + 2; nd = int((d1 - d0) / step_mm) + 2
    tu = T.linspace(u0, u1, nu, device=dev); tv = T.linspace(v0, v1, nv, device=dev)
    td = d0 + (T.arange(nd, device=dev, dtype=T.float32) + 0.5) * step_mm
    pts = (td.view(-1,1,1,1)*d.view(1,1,1,3) + tu.view(1,-1,1,1)*u.view(1,1,1,3) + tv.view(1,1,-1,1)*v.view(1,1,1,3))
    inv_w = 2.0/max(sx*(nx-1),1e-6); inv_h = 2.0/max(sy*(ny-1),1e-6); inv_d = 2.0/max(sz*(nz-1),1e-6)
    g = T.stack([(pts[...,0]-ox)*inv_w-1, (pts[...,1]-oy)*inv_h-1, (pts[...,2]-oz)*inv_d-1], dim=-1)[None]
    del pts
    rho = FF.grid_sample(dens5, g, mode="bilinear", align_corners=True, padding_mode="zeros")[0,0]
    del g
    w_src_g = (T.cumsum(rho, dim=0) * (step_mm/10.0)).half()
    hit = (rho > skin_thr).float()
    crossed = (T.cumsum(hit, dim=0) > 0).float()
    w_skin_g = (T.cumsum(rho * crossed, dim=0) * (step_mm/10.0)).half()
    ent_g = crossed.half()
    del rho, hit, crossed
    T.cuda.empty_cache()
    return (u, v, d), (u0,u1,v0,v1,d0,nd,step_mm), w_src_g, w_skin_g, ent_g


def wepl_beam_sample(basis, ext, w_src_g, w_skin_g, ent_g, coords):
    """Sample the fp16 BEV grids at coords -> (w_src, w_skin, entered) fp32."""
    import torch as T, torch.nn.functional as FF
    u, v, d = basis; u0,u1,v0,v1,d0,nd,step_mm = ext
    P = coords.reshape(-1, 3)
    qu = ((P @ u) - u0) / max(u1-u0,1e-9) * 2 - 1
    qv = ((P @ v) - v0) / max(v1-v0,1e-9) * 2 - 1
    qd = (((P @ d) - d0) / step_mm - 0.5) / max(nd-1,1) * 2 - 1
    q = T.stack([qv, qu, qd], dim=-1).view(1,-1,1,1,3)
    del P, qu, qv, qd
    outs = []
    for gimg in (w_src_g, w_skin_g, ent_g):
        o = FF.grid_sample(gimg[None,None].float(), q, mode="bilinear", align_corners=True,
                           padding_mode="border").view(-1).reshape(coords.shape[:-1])
        outs.append(o)
    del q
    return outs[0], outs[1], outs[2]
