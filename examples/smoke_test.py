#!/usr/bin/env python3
"""Five-minute installation check on a synthetic phantom. No challenge data, no GPU needed.

    python examples/smoke_test.py            # CPU
    python examples/smoke_test.py --device cuda

It builds a small water phantom with a lung-like low-density insert and a bone-like insert,
defines one photon control point (a rectangular MLC aperture), runs the analytical physics
operator, pushes the channels through an untrained DoseUNet3D, and prints shape/range checks.
The predicted dose is meaningless (random weights) — the point is that the physics operator, the
network, and their plumbing all work in your environment.

What you should see: the radiological depth increases along the beam, the fluence is confined to
the aperture, the naive prior peaks a few centimetres deep, and the network returns a
non-negative volume of the same shape.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import torch

from doserad.io.mha import Volume
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.channels import photon_channels
from doserad.physics.machine import PhotonMachine


def build_phantom(nz=48, ny=96, nx=96, spacing=(2.0, 2.0, 2.0)) -> Volume:
    """Water box with a low-density (lung-like) slab and a dense (bone-like) block, in HU."""
    hu = np.full((nz, ny, nx), -1000.0, np.float32)          # air around the phantom
    hu[6:nz - 6, 12:ny - 12, 12:nx - 12] = 0.0               # water
    hu[16:32, 30:66, 30:66] = -750.0                         # lung-like insert
    hu[20:26, 44:52, 20:28] = 900.0                          # bone-like insert
    return Volume(array=hu, spacing=spacing, origin=(0.0, 0.0, 0.0),
                  direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))


def build_machine() -> PhotonMachine:
    """A minimal, physically plausible machine model (not a clinical beam model)."""
    return PhotonMachine(
        sad_mm=1000.0, num_leaf_pairs=64, leaf_thickness_mm=5.0,
        jaw_x_mm=(-100.0, 100.0), jaw_y_mm=(-100.0, 100.0),
        source_plane_distance_mm=1000.0, virtual_source_distance_mm=1000.0,
        spectrum_mev=(6.0,), spectrum_weight=(1.0,),
        hu_anchors=((-1000.0, 0.0), (0.0, 1.0), (1000.0, 1.9)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    dev = a.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        dev = "cpu"

    image = build_phantom()
    machine = build_machine()
    nz, ny, nx = image.array.shape
    iso = [nx * image.spacing[0] / 2, ny * image.spacing[1] / 2, nz * image.spacing[2] / 2]

    # One control point: a 4 cm x 4 cm rectangular opening centred on the isocentre axis.
    n = machine.num_leaf_pairs
    left = np.full(n, 0.0, np.float64)
    right = np.full(n, 0.0, np.float64)
    lo, hi = n // 2 - 4, n // 2 + 4
    left[lo:hi], right[lo:hi] = -20.0, 20.0          # mm, open in the central leaf pairs

    print(f"phantom {nz}x{ny}x{nx} @ {image.spacing} mm, isocentre {np.round(iso, 1)} mm")
    print("building analytical physics channels ...", flush=True)
    ch = photon_channels(image=image, machine=machine, iso_xyz=iso, gantry_deg=0.0,
                         mlc_left=left, mlc_right=right)
    ch = np.asarray(ch, dtype=np.float32)
    names = ["density", "radiological depth", "primary fluence", "dist-to-CAX", "source distance"]
    print(f"channels: {ch.shape}")
    for i, nm in enumerate(names[:ch.shape[0]]):
        c = ch[i]
        print(f"  {nm:>18}: min {c.min():9.3f}  max {c.max():9.3f}  mean {c.mean():9.3f}")

    open_frac = float((ch[2] > 0).mean())
    print(f"  fluence covers {open_frac * 100:.1f}% of the volume (a small aperture should be a "
          f"few percent)")

    print("\nrunning an untrained DoseUNet3D forward (random weights) ...", flush=True)
    net = DoseUNet3D(in_ch=ch.shape[0], base=16, levels=3, bottleneck="dilated").to(dev).eval()
    x = torch.from_numpy(ch)[None].to(dev)
    pad = [(0, (8 - s % 8) % 8) for s in x.shape[2:]]
    x = torch.nn.functional.pad(x, [v for p in reversed(pad) for v in p])
    with torch.no_grad():
        y = net(x, torch.zeros(1, dtype=torch.long, device=dev))
    y = y[0, 0, :nz, :ny, :nx]
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  network {params:.2f} M params on {dev}; output {tuple(y.shape)}, "
          f"min {y.min():.4f}, max {y.max():.4f}")

    ok = (y.shape == (nz, ny, nx)) and bool((y >= 0).all()) and 0.0 < open_frac < 0.5
    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED")
    print("Next: docs/WHICH_FILE.md for the real recipes, docs/INFERENCE.md to run a trained "
          "container.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
