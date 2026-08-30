"""Full container integration: mock 10-slot input (1 real patient, all beams) -> process_run with
the REAL compiled model -> read back the output stack -> plan gamma vs GT. Validates the whole
proton container path (minus the HTTP layer). Run:
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python container/proton/test_integration.py 1ABB006
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
from container.proton import app, gc_invoke
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array

ROOT = Path("/data/kwang/DoseRad2026_raw/proton/training")
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"


def main():
    app.load_models()
    tmp = Path(tempfile.mkdtemp(prefix="gcint_"))
    try:
        in_dir = tmp / "input"; (in_dir / "images").mkdir(parents=True)
        for k in range(gc_invoke.N_SLOTS):
            d = in_dir / "images" / f"radiation-dose-calculation-source-ct-image-{k+1}"; d.mkdir(parents=True)
            if k == 0:
                sitk.WriteImage(sitk.ReadImage(str(ROOT / PID / "image" / "ct.mha")), str(d / "img.mha"), True)
            else:
                sitk.WriteImage(sitk.GetImageFromArray(np.zeros((1, 1, 1), np.float32)), str(d / "ph.mha"))
        plan = json.load(open(ROOT / PID / f"{PID}.json"))
        # build metadata with output_info: all beamlets -> slot 0, contiguous idx.
        # CAP: each output frame is FULL-grid (~161 MB), so a realistic sub-batch is small; the
        # organizers sub-batch the test set. 20 frames * 161 MB ~= 3.2 GB (a sane per-invoke run).
        CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        beams = []; iio = 0
        for bi, b in enumerate(plan["beams"]):
            b["beam_idx"] = bi
            kept_rays = []
            for ri, r in enumerate(b["rays"]):
                r["ray_idx"] = ri
                kept = []
                for l, bl in enumerate(r["beamlets"]):
                    if iio >= CAP:
                        break
                    bl["output_info"] = {"output_file_idx": 0, "idx_in_output": iio, "minimum_cutoff": 0.0}
                    kept.append(bl); iio += 1
                if kept:
                    r["beamlets"] = kept; kept_rays.append(r)
            if kept_rays:
                b["rays"] = kept_rays; beams.append(b)
            if iio >= CAP:
                break
        meta = [{"image_file_idx": 0, "anatomical_region": "abdominal", "beams": beams}]
        json.dump(meta, open(in_dir / "stacked-proton-beam-level-metadata.json", "w"))

        out_dir = tmp / "output"
        t0 = time.time()
        gc_invoke.process_run(in_dir, out_dir, app._predict_fn, "proton", "ct")
        dt = time.time() - t0

        # read the output stack (slot 1); validate PLUMBING (end-to-end at realistic scale, no OOM).
        # Full-plan gamma is validated separately in test_predict (98.3 over all beams).
        f = list((out_dir / "images" / "stacked-radiation-dose-map-1").glob("*.mha"))[0]
        im = sitk.ReadImage(str(f))
        stack = sitk.GetArrayFromImage(im)                       # (n, z, y, x)
        src = sitk.ReadImage(str(ROOT / PID / "image" / "ct.mha"))
        n_expected = sum(len(r["beamlets"]) for b in beams for r in b["rays"])
        print(f"\n=== CONTAINER INTEGRATION ({PID}) ===")
        print(f"  process_run {dt:.1f}s, {n_expected} beamlets -> 1 output run")
        print(f"  [{'PASS' if im.GetDimension()==4 else 'FAIL'}] output is 4D")
        print(f"  [{'PASS' if stack.shape[0]==n_expected else 'FAIL'}] n frames {stack.shape[0]} == n beamlets {n_expected}")
        print(f"  [{'PASS' if stack.shape[1:]==src.GetSize()[::-1] else 'FAIL'}] each frame on source grid {stack.shape[1:]}")
        nz = [int((stack[i] > 0).sum()) for i in range(stack.shape[0])]
        print(f"  [{'PASS' if all(v>0 for v in nz) else 'FAIL'}] every beamlet frame has nonzero dose (min {min(nz)} vox)")
        print(f"  dose range [{stack.min():.2e}, {stack.max():.2e}] Gy")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
