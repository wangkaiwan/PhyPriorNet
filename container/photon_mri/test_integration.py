"""Photon-MRI container integration: mock 10-slot MRI input (source-mri-image folders) -> real
gc_invoke.process_run with the compiled model -> read back the 4D output stack. Validates the
modality='mri' I/O contract (slot folder naming, JoinSeries, grid, nonzero). CPs capped for speed.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python container/photon_mri/test_integration.py 1ABB006 30
"""
from __future__ import annotations
import json, os, sys, shutil, tempfile, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/cv/se_photonmri_f0.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/se_photonmri_f0/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json")
from pathlib import Path
import numpy as np, SimpleITK as sitk
from container.photon_mri import app
from container.proton import gc_invoke

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def main():
    app.load_models()
    tmp = Path(tempfile.mkdtemp(prefix="gcmri_"))
    try:
        in_dir = tmp / "input"; (in_dir / "images").mkdir(parents=True)
        for k in range(gc_invoke.N_SLOTS):
            d = in_dir / "images" / f"radiation-dose-calculation-source-mri-image-{k+1}"; d.mkdir(parents=True)
            if k == 0:
                sitk.WriteImage(sitk.ReadImage(f"{ROOT}/{PID}/image/mr.mha"), str(d / "img.mha"), True)
            else:
                sitk.WriteImage(sitk.GetImageFromArray(np.zeros((1, 1, 1), np.float32)), str(d / "ph.mha"))
        plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
        beams = []; iio = 0
        for bi, b in enumerate(plan["beams"]):
            b["beam_idx"] = bi; kept = []
            for ci, cp in enumerate(b["control_points"]):
                if iio >= CAP:
                    break
                cp["output_info"] = {"output_file_idx": 0, "idx_in_output": iio, "minimum_cutoff": 0.0}
                cp["_ci"] = ci; kept.append(cp); iio += 1
            if kept:
                b["control_points"] = kept; beams.append(b)
            if iio >= CAP:
                break
        meta = [{"image_file_idx": 0, "anatomical_region": "abdominal", "beams": beams}]
        json.dump(meta, open(in_dir / "stacked-photon-beam-level-metadata.json", "w"))

        out_dir = tmp / "output"; t0 = time.time()
        gc_invoke.process_run(in_dir, out_dir, app._predict_fn, "photon", "mri")
        dt = time.time() - t0
        f = list((out_dir / "stacked-radiation-dose-map-1").glob("*.mha"))[0]
        im = sitk.ReadImage(str(f)); stack = sitk.GetArrayFromImage(im)
        src = sitk.ReadImage(f"{ROOT}/{PID}/image/mr.mha")
        n_exp = sum(len(b["control_points"]) for b in beams)
        print(f"\n=== PHOTON-MRI CONTAINER INTEGRATION ({PID}, cap {CAP}) ===")
        print(f"  process_run {dt:.1f}s, {n_exp} CPs -> 1 output run")
        print(f"  [{'PASS' if im.GetDimension()==4 else 'FAIL'}] output is 4D")
        print(f"  [{'PASS' if stack.shape[0]==n_exp else 'FAIL'}] n frames {stack.shape[0]} == n CPs {n_exp}")
        print(f"  [{'PASS' if stack.shape[1:]==src.GetSize()[::-1] else 'FAIL'}] frame grid {stack.shape[1:]} == source {src.GetSize()[::-1]}")
        nz = [int((stack[i] > 0).sum()) for i in range(stack.shape[0])]
        print(f"  [{'PASS' if all(v>0 for v in nz) else 'FAIL'}] every CP frame nonzero (min {min(nz)} vox)")
        # all 10 output slots present
        slots_ok = all((out_dir / f"stacked-radiation-dose-map-{s+1}").glob("*.mha") for s in range(10))
        print(f"  [{'PASS' if slots_ok else 'FAIL'}] all 10 output slots written")
        print(f"  dose range [{stack.min():.2e}, {stack.max():.2e}] Gy")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
