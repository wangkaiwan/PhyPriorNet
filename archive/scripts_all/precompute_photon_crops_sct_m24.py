"""sCT-DOMAIN photon channel cache (photon-MRI b32 domain-distill, user OK 2026-08-26).

The CT-trained b32 student failed the MRI gate by −4.07γ (never saw sCT channels). Fix = distill in
the deploy domain: for all 75 patients, run the DEPLOYED photon-MRI front-end (clf_whole coarse +
m16_mm E2E synth -> sCT -> density, density_direct=False) and recompute the 5 skin-entry channels
from that density on the SAME m24-cache bboxes (GT dose copied) -> a drop-in cache_dir for
distill_dose_photon. Honest test-time setup: MR in, no GT-CT leakage.

Out: cache/crops/photon_skinentry_sct_m24/<pid>/<B>_<CP>.npz {channels(5,fp16), dose, bbox, dose_max}
DOSERAD_SHARD=k/N; DOSERAD_FORCE=1. Usage: CUDA_VISIBLE_DEVICES=N python <me> [pids...]
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, "scripts")
import numpy as np
import SimpleITK as sitk
import torch
import yaml

from doserad.beam.parse import load_photon_plan
from doserad.data.crop import crop_to_bbox
from doserad.io.mha import load_mha
from doserad.physics.channels_skinentry import photon_channels_skinentry
from doserad.physics.machine import load_photon_machine
from container.mri_synth import load_classifier, synth_density

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
SRC = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")
OUT = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_sct_m24")
E2E_STATE = "/home/kaiwang/doserad2026_workdir/runs/mm_ftm16_photonmri/state.pt"
E2E_CFG = "configs/experiments/all75/m24S2_p4_mmB.yaml"
CLF = "/data/kwang/sct_classify_runs/clf_whole/best.pt"
DEV = "cuda"
FORCE = bool(os.environ.get("DOSERAD_FORCE"))
MACHINE = load_photon_machine(f"{ROOT}/beam_parameters.json")


@torch.no_grad()
def main(pids):
    from train_dose_e2e import E2E
    cfg = yaml.safe_load(open(E2E_CFG))
    net = E2E(cfg).to(DEV).eval()
    sd = torch.load(E2E_STATE, map_location=DEV)
    net.load_state_dict(sd.get("ema", sd.get("model")))
    clf = load_classifier(CLF, DEV)
    if not pids:
        pids = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    shard = os.environ.get("DOSERAD_SHARD")
    if shard:
        k, N = (int(x) for x in shard.split("/"))
        pids = [p for i, p in enumerate(pids) if i % N == k]
    print(f"[sct-cache] {len(pids)} pids -> {OUT}", flush=True)
    # zlib in np.savez_compressed releases the GIL, so a thread pool parallelizes the compression
    # that dominates wall time (~4.3 G/patient at ~0.3 GB/s single-thread). Writes stay atomic
    # (unique tmp name per file via a counter, os.replace).
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(int(os.environ.get("DOSERAD_WRITERS", "6")))

    def _save(tpath, dst, ch_c, dose, bbox, dose_max):
        np.savez_compressed(tpath, channels=ch_c, dose=dose,
                            bbox=np.asarray(bbox, np.int32), dose_max=dose_max)
        os.replace(tpath, dst)

    for pid in pids:
        t0 = time.time()
        pdir = Path(ROOT) / pid
        mr = sitk.ReadImage(str(pdir / "image" / "mr.mha"))
        density, _ = synth_density(mr, clf, net, DEV, density_direct=False,
                                   hu_anchors=MACHINE.hu_anchors)
        density = density.astype(np.float32)
        ct = load_mha(pdir / "image" / "mr.mha")     # grid/spacing/origin carrier (MR grid == CT grid)
        plan = load_photon_plan(pdir / f"{pid}.json")
        (OUT / pid).mkdir(parents=True, exist_ok=True)
        n = 0
        futs = []
        for beam in plan.beams:
            for cp in beam.control_points:
                fn = f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
                if (OUT / pid / fn).exists() and not FORCE:
                    continue
                zsrc = np.load(SRC / pid / fn)
                bbox = tuple(int(v) for v in zsrc["bbox"])
                chans = photon_channels_skinentry(
                    image=ct, machine=MACHINE, iso_xyz=beam.iso_center,
                    gantry_deg=cp.gantry_angle,
                    mlc_left=np.asarray(cp.mlc_left_int_mm),
                    mlc_right=np.asarray(cp.mlc_right_int_mm),
                    density_override=density)
                ch_c = crop_to_bbox(chans, bbox).astype(np.float16)
                tpath = (OUT / pid / fn).with_suffix(f".{os.getpid()}.{n}.tmp.npz")
                futs.append(pool.submit(_save, tpath, OUT / pid / fn,
                                        ch_c, zsrc["dose"], bbox, zsrc["dose_max"]))
                n += 1
                if len(futs) >= 64:            # bound in-flight arrays (RAM)
                    futs.pop(0).result()
        for f in futs:
            f.result()
        print(f"  {pid}: {n} CPs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
