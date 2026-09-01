"""Break down per-dosemap inference time into PHYSICS-CHANNELS vs NETWORK, to
decide where to optimize (the measured ~540 ms/dosemap is thin on A10G).

Times, per CP (after warmup), with proper CUDA synchronization:
  - photon_channels(crop_margin)  [physics: projection + open_mask + bbox rdepth + per-voxel channels]
  - normalize_channels            [cpu]
  - net forward                   [the 3D U-Net]
Reports mean ms per phase + the crop sizes (bbox volume) seen.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from doserad.beam.parse import load_photon_plan
from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.channels import photon_channels
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.inference.pipeline import _build_coords, _flush_batch, _normalize_gpu

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="1ABB006")
    ap.add_argument("--ckpt", default="/home/kaiwang/doserad2026_workdir/runs/v6_photon_ct_naive/best.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--in-ch", type=int, default=6)
    ap.add_argument("--add-naive", action="store_true", default=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pdir = Path(ROOT) / args.pid
    image = load_mha(pdir / "image" / "ct.mha")
    plan = load_photon_plan(pdir / f"{args.pid}.json")
    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    net = DoseUNet3D(in_ch=args.in_ch, base=args.base_ch, levels=4).to(dev).eval()
    st = torch.load(args.ckpt, map_location=dev)
    net.load_state_dict(st.get("ema", st.get("model")))

    density = hu_to_density(image.array, machine.hu_anchors)
    coords = _build_coords(image, dev)
    full_shape = image.array.shape

    cps = [(b, cp) for b in plan.beams for cp in b.control_points][: args.n + 2]
    t_chan = []; t_norm = []; t_net = []; sizes = []
    for i, (beam, cp) in enumerate(cps):
        _sync(); t0 = time.time()
        crop, bbox = photon_channels(
            image=image, machine=machine, iso_xyz=beam.iso_center,
            gantry_deg=cp.gantry_angle,
            mlc_left=np.asarray(cp.mlc_left_int_mm),
            mlc_right=np.asarray(cp.mlc_right_int_mm),
            density_override=density, coords=coords, crop_margin=8,
            return_tensor=True)
        _sync(); t1 = time.time()
        cn = _normalize_gpu(crop, args.add_naive)
        _sync(); t2 = time.time()
        out = {}
        _flush_batch(net, [((0, 0), cn, bbox)], dev, full_shape, out, 0)
        _sync(); t3 = time.time()
        if i >= 2:                       # skip first 2 (warmup)
            t_chan.append(t1 - t0); t_norm.append(t2 - t1); t_net.append(t3 - t2)
            sizes.append(int(cn.shape[-3]) * int(cn.shape[-2]) * int(cn.shape[-1]))

    def stat(a):
        a = np.array(a) * 1000.0
        return f"{a.mean():.0f} ± {a.std():.0f} ms"
    tot = (np.array(t_chan) + np.array(t_norm) + np.array(t_net)).mean() * 1000
    print(f"profiled {len(t_chan)} CPs on {dev}, crop voxels mean "
          f"{np.mean(sizes)/1e6:.2f}M (max {np.max(sizes)/1e6:.2f}M)")
    print(f"  physics_channels : {stat(t_chan)}  ({np.mean(t_chan)/(tot/1000)*100:.0f}%)")
    print(f"  normalize (gpu)  : {stat(t_norm)}")
    print(f"  net forward      : {stat(t_net)}  ({np.mean(t_net)/(tot/1000)*100:.0f}%)")
    print(f"  TOTAL per CP     : {tot:.0f} ms")


if __name__ == "__main__":
    main()
