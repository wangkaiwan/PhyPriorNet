"""Overlapped writer == serial writer, exactly.

The overlapped path (DOSERAD_OVERLAP_WRITE=1) streams each beamlet to a background per-slot thread
as soon as it is computed, instead of collecting every crop and writing at the end. That reorders
*when* frames are produced, so the thing to prove is that it does not reorder *where* they land:
run the same patient through both paths with the real model and require bit-identical stacks.

Also checks the multi-slot case, since each slot gets its own writer thread.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n doserad python container/proton/test_overlap_equiv.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/all75_r2_ft/state.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
from container.proton import app, gc_invoke

ROOT = Path("/data/kwang/DoseRad2026_raw/proton/training")
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 24
N_SLOTS_USED = 2          # spread the beamlets over two output slots -> two writer threads


def build_input(tmp: Path) -> Path:
    in_dir = tmp / "input"
    (in_dir / "images").mkdir(parents=True)
    for k in range(gc_invoke.N_SLOTS):
        d = in_dir / "images" / f"radiation-dose-calculation-source-ct-image-{k+1}"
        d.mkdir(parents=True)
        if k == 0:
            sitk.WriteImage(sitk.ReadImage(str(ROOT / PID / "image" / "ct.mha")), str(d / "img.mha"), True)
        else:
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((1, 1, 1), np.float32)), str(d / "ph.mha"))

    plan = json.load(open(ROOT / PID / f"{PID}.json"))
    beams, taken, per_slot = [], 0, [0] * N_SLOTS_USED
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        kept_rays = []
        for ri, r in enumerate(b["rays"]):
            r["ray_idx"] = ri
            kept = []
            for bl in r["beamlets"]:
                if taken >= CAP:
                    break
                s = taken % N_SLOTS_USED          # interleave slots so ordering is non-trivial
                bl["output_info"] = {"output_file_idx": s, "idx_in_output": per_slot[s],
                                     "minimum_cutoff": 1e-6}
                per_slot[s] += 1
                kept.append(bl)
                taken += 1
            if kept:
                r["beamlets"] = kept
                kept_rays.append(r)
        if kept_rays:
            b["rays"] = kept_rays
            beams.append(b)
        if taken >= CAP:
            break
    json.dump([{"image_file_idx": 0, "anatomical_region": "abdominal", "beams": beams}],
              open(in_dir / "stacked-proton-beam-level-metadata.json", "w"))
    return in_dir, taken, per_slot


def run(in_dir: Path, out_dir: Path, overlap: bool):
    gc_invoke._OVERLAP_WRITE = overlap
    st = {}
    t0 = time.time()
    slots = gc_invoke.process_run(in_dir, out_dir, app._predict_fn, "proton", "ct", stats=st)
    return time.time() - t0, sum(len(v) for v in slots.values()), st


def main():
    app.load_models()
    tmp = Path(tempfile.mkdtemp(prefix="ovl_", dir="/data/kwang"))
    try:
        in_dir, n_beamlets, per_slot = build_input(tmp)
        print(f"=== OVERLAP EQUIVALENCE ({PID}) — {n_beamlets} beamlets over "
              f"{N_SLOTS_USED} slots {per_slot} ===")

        dt_s, n_s, st_s = run(in_dir, tmp / "serial", overlap=False)
        dt_o, n_o, st_o = run(in_dir, tmp / "overlap", overlap=True)
        print(f"  serial     {dt_s:6.2f}s  ({n_s} dosemaps)")
        print(f"  overlapped {dt_o:6.2f}s  ({n_o} dosemaps)  missing={st_o.get('missing')}")

        ok = True
        for s in range(N_SLOTS_USED):
            sub = f"images/stacked-radiation-dose-map-{s+1}/output.mha"
            a = sitk.ReadImage(str(tmp / "serial" / sub))
            b = sitk.ReadImage(str(tmp / "overlap" / sub))
            same_geom = (a.GetSize() == b.GetSize() and np.allclose(a.GetSpacing(), b.GetSpacing())
                         and np.allclose(a.GetOrigin(), b.GetOrigin())
                         and np.allclose(a.GetDirection(), b.GetDirection()))
            aa, bb = sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b)
            same_vox = np.array_equal(aa, bb)
            # per-frame check too: a swap between two frames would still pass a set-wise compare
            per_frame = [bool(np.array_equal(aa[i], bb[i])) for i in range(aa.shape[0])]
            print(f"  [{'PASS' if same_geom else 'FAIL'}] slot {s+1} geometry identical {b.GetSize()}")
            print(f"  [{'PASS' if same_vox else 'FAIL'}] slot {s+1} voxels bit-identical")
            print(f"  [{'PASS' if all(per_frame) else 'FAIL'}] slot {s+1} every frame in the same "
                  f"position ({sum(per_frame)}/{len(per_frame)})")
            ok &= same_geom and same_vox and all(per_frame)
            nz = [int((bb[i] > 0).sum()) for i in range(bb.shape[0])]
            print(f"  [{'PASS' if all(v > 0 for v in nz) else 'FAIL'}] slot {s+1} no zero-filled "
                  f"frame (min {min(nz)} vox)")
            ok &= all(v > 0 for v in nz)

        for s in range(N_SLOTS_USED, gc_invoke.N_SLOTS):
            b = sitk.ReadImage(str(tmp / "overlap" / f"images/stacked-radiation-dose-map-{s+1}/output.mha"))
            ok &= b.GetSize() == (1, 1, 1)
        print(f"  [{'PASS' if ok else 'FAIL'}] unused slots are 1x1x1 placeholders")
        print("\nOVERLAP EQUIVALENCE: " + ("ALL PASS" if ok else "FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
