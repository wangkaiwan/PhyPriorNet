"""Acceleration sprint — photon deploy build profile (mirror of profile_proton_build).

Per CP the deployed path: photon_channels (density/rdepth/fluence/dist/src channels, GPU raytrace)
-> _normalize_gpu (+ naive prior) -> batched forward (<=8, shared padding). Times each component
over N CPs of one patient; forward timed per-batch and amortized per CP.

Run (GPU1): CUDA_VISIBLE_DEVICES=1 conda run -n doserad python -u accel/profile_photon_build.py
"""
from __future__ import annotations

import sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doserad.physics.channels import photon_channels                 # noqa: E402
from doserad.physics.density import hu_to_density                    # noqa: E402
from doserad.physics.machine import load_photon_machine              # noqa: E402
from doserad.inference.pipeline import _normalize_gpu, _build_coords # noqa: E402
from doserad.beam.parse import load_photon_plan                      # noqa: E402
from doserad.io.mha import load_mha                                  # noqa: E402
from doserad.data.dataset import DOSE_SCALE                          # noqa: E402
from doserad.model.unet3d import DoseUNet3D                          # noqa: E402

DEV = "cuda"
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
PID = "1ABB006"
N = 64


def sync():
    torch.cuda.synchronize()


def main():
    cfg = yaml.safe_load(open("configs/experiments/cv/ftg_skinentry_photonct_f0.yaml"))
    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(DEV).eval()
    net.load_state_dict(torch.load(
        "/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt",
        map_location=DEV)["ema"])
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    ct = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    plan = load_photon_plan(Path(f"{ROOT}/{PID}/{PID}.json"))
    density = hu_to_density(ct.array, machine.hu_anchors)
    coords = _build_coords(ct, DEV)

    cps = [(beam, cp) for beam in plan.beams for cp in beam.control_points][: N + 6]
    T = defaultdict(float); n = 0
    buf = []
    factor, infer_batch = 8, 8

    def flush():
        nonlocal buf
        if not buf:
            return 0.0
        t0 = time.perf_counter()
        dims = [c.shape[-3:] for c in buf]
        D = -(-max(d[0] for d in dims) // factor) * factor
        H = -(-max(d[1] for d in dims) // factor) * factor
        W = -(-max(d[2] for d in dims) // factor) * factor
        xb = torch.zeros((len(buf), buf[0].shape[0], D, H, W), dtype=torch.float32, device=DEV)
        for i, c in enumerate(buf):
            d, h, w = c.shape[-3:]; xb[i, :, :d, :h, :w] = c.to(DEV)
        with torch.no_grad(), torch.autocast("cuda"):
            _ = net(xb, torch.zeros(len(buf), dtype=torch.long, device=DEV))
        sync()
        k = len(buf); buf = []
        return (time.perf_counter() - t0), k

    fw_time = 0.0; fw_cps = 0
    for i, (beam, cp) in enumerate(cps):
        t0 = time.perf_counter()
        crop, bbox = photon_channels(image=ct, machine=machine, iso_xyz=beam.iso_center,
                                     gantry_deg=cp.gantry_angle,
                                     mlc_left=np.asarray(cp.mlc_left_int_mm),
                                     mlc_right=np.asarray(cp.mlc_right_int_mm),
                                     density_override=density, coords=coords,
                                     crop_margin=8, return_tensor=True)
        sync(); t1 = time.perf_counter()
        crop = _normalize_gpu(crop, True)
        sync(); t2 = time.perf_counter()
        if i >= 6:
            T["photon_channels"] += t1 - t0; T["normalize_naive"] += t2 - t1; n += 1
        d, h, w = crop.shape[-3:]
        if buf:
            dims = [c.shape[-3:] for c in buf] + [(d, h, w)]
            D = max(x[0] for x in dims); H = max(x[1] for x in dims); W = max(x[2] for x in dims)
            if (len(buf) + 1) * D * H * W > 2_500_000:
                ft, k = flush()
                if i >= 6:
                    fw_time += ft; fw_cps += k
        buf.append(crop)
        if len(buf) >= infer_batch:
            ft, k = flush()
            if i >= 6:
                fw_time += ft; fw_cps += k
    if buf:
        ft, k = flush(); fw_time += ft; fw_cps += k

    per_fw = fw_time / max(fw_cps, 1)
    tot = sum(T.values()) / n + per_fw
    print(f"\n=== PHOTON deploy build profile ({PID}, {n} CPs, ms/CP) ===")
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s}: {v / n * 1e3:7.1f} ms  ({v / n / tot * 100:4.1f}%)")
    print(f"  forward(b<=8)   : {per_fw * 1e3:7.1f} ms  ({per_fw / tot * 100:4.1f}%)")
    print(f"  TOTAL           : {tot * 1e3:7.1f} ms/CP")


if __name__ == "__main__":
    main()
