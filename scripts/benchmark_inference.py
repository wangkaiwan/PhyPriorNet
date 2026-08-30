"""Benchmark per-dosemap inference wall-clock on a real patient.

The challenge gates on ([total runtime] - [fixed startup]) / [num beams], i.e. the
MARGINAL per-dosemap cost, with model/docker/GPU load EXCLUDED (organizers estimate
it by regressing total time against input size). We mirror that: time the full
photon_inference at several CP counts N and fit a line — the SLOPE is the
startup-excluded per-dosemap time (model load + per-image setup cancel out), the
intercept is the fixed overhead.

Usage: conda run -n doserad python scripts/benchmark_inference.py \
           [--pid 1ABB006] [--ns 8,24,48] [--ckpt PATH] [--base-ch 48] [--in-ch 5]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from doserad.beam.parse import load_photon_plan
from doserad.inference.pipeline import photon_inference
from doserad.io.mha import load_mha
from doserad.physics.machine import load_photon_machine

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"


def _truncate(plan, n):
    kept = 0
    for beam in plan.beams:
        if kept >= n:
            beam.control_points.clear(); continue
        if kept + len(beam.control_points) > n:
            beam.control_points[:] = beam.control_points[:n - kept]
        kept += len(beam.control_points)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="1ABB006")
    ap.add_argument("--ns", default="8,24,48", help="comma-sep CP counts to time")
    ap.add_argument("--ckpt",
        default="/home/kaiwang/doserad2026_workdir/runs/v6_photon_ct_naive/best.pt")
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--in-ch", type=int, default=5)
    ap.add_argument("--add-naive", action="store_true")
    ap.add_argument("--bottleneck", default="plain")
    ap.add_argument("--infer-batch", type=int, default=8)
    args = ap.parse_args()
    ns = sorted(int(x) for x in args.ns.split(","))
    pdir = Path(ROOT) / args.pid
    image = load_mha(pdir / "image" / "ct.mha")
    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    kw = dict(modality="ct", base_ch=args.base_ch, in_ch=args.in_ch,
              add_naive=args.add_naive, bottleneck=args.bottleneck, infer_batch=args.infer_batch)

    def run(n):
        plan = _truncate(load_photon_plan(pdir / f"{args.pid}.json"), n)
        t0 = time.time()
        out = photon_inference(image, plan, args.ckpt, machine, **kw)
        return time.time() - t0, len(out)

    print(f"warmup (load + caches)..."); run(max(ns))
    xs, ts = [], []
    for n in ns:
        dt, got = run(n)
        xs.append(got); ts.append(dt)
        print(f"  N={got:3d} CPs : {dt:6.2f}s total ({dt/got*1000:.0f} ms/CP incl. fixed)")
    xs = np.array(xs, float); ts = np.array(ts, float)
    slope, intercept = np.polyfit(xs, ts, 1)         # ts = slope*N + intercept
    per = slope * 1000.0
    print(f"\nfit: per-dosemap (slope) = {per:.0f} ms ; fixed overhead = {intercept*1000:.0f} ms")
    print(f"target <1000 ms/dosemap. headroom: {1000.0/per:.1f}x")
    a10g = per * 1.5
    print(f"A10G estimate (~1.5x): {a10g:.0f} ms/dosemap -- "
          f"{'OK' if a10g < 1000 else 'OVER BUDGET (next: torch.compile/TensorRT, smaller BEV grid)'}")


if __name__ == "__main__":
    main()
