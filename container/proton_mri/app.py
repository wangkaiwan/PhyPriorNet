"""Grand-Challenge invoke server for the PROTON-MRI dose algorithm.

MRI -> (clf_whole coarse + E2E synth, density_direct) sCT density, RESAMPLED to the source (native
proton) grid -> ray-centric build (fused WEPL + skin-entry PB) -> compiled dose net. Reuses the
validated proton deploy path (container.proton.predict.predict_beams + geom_bbox); only the density
source changes (synth instead of real-CT HU). Density built ONCE per source image.
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from container.proton import gc_invoke
from container.proton.predict import predict_beams
from container.mri_synth import synth_density, load_classifier

def _resolve(env, default):
    """Env-configured path, but a GC separate 'model' upload (extracted to /opt/ml/model/) wins —
    lets us swap weights (e.g. all-75 finals) without rebuilding/re-uploading the image."""
    p = os.environ.get(env, default)
    ml = Path("/opt/ml/model") / Path(p).name
    return str(ml) if ml.exists() else p

CFG = _resolve("DOSERAD_CONFIG", "/opt/algorithm/weights/se_protonmri_f0.yaml")
WEIGHTS = _resolve("DOSERAD_WEIGHTS", "/opt/algorithm/weights/proton_mri.pt")
CLF = _resolve("DOSERAD_CLF", "/opt/algorithm/weights/clf_whole.pt")
MACHINE = _resolve("DOSERAD_MACHINE", "/opt/algorithm/weights/beam_parameters.json")
INPUT = Path(os.environ.get("DOSERAD_INPUT", "/input"))
OUTPUT = Path(os.environ.get("DOSERAD_OUTPUT", "/output"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# Proton deliberately does NOT torch.compile. Its invoke is write-bound with a large margin
# (measured: 95.7 s compute against a 203.8 s write floor), so even eager -- ~1.39x slower on the
# dose net -- stays completely hidden and invoke time does not move. Meanwhile the compile costs
# ~100 s of STARTUP, and startup counts against SageMaker's MaxRuntimeInSeconds, which is what
# actually killed a proton-MRI job (275 s invoke + ~100 s startup). Photon keeps compile: its
# invoke is only ~46 s, its compile win is ~3x, and dropping it would make it compute-bound.
# DOSERAD_COMPILE=1 restores the old behaviour.
_COMPILE = os.environ.get("DOSERAD_COMPILE", "0") == "1"
_STATE = {"ready": False, "timings": {}}


def load_models():
    from train_dose_e2e import E2E
    cfg = yaml.safe_load(open(CFG))
    machine = load_photon_machine(MACHINE)
    pm = ProtonMachineData(device=DEV)
    net = E2E(cfg).to(DEV).eval()
    sd = torch.load(WEIGHTS, map_location=DEV)
    net.load_state_dict(sd.get("ema", sd.get("model")))
    clf = load_classifier(CLF, DEV)
    if DEV == "cuda":
        # net.synth is deliberately NOT compiled. It runs ONCE PER SOURCE IMAGE, not per
        # beamlet, so compilation buys almost nothing -- but every new input shape costs an
        # inductor recompile INSIDE the ranked invoke, and source grids differ per patient. A
        # platform proton-MRI job carrying 6 distinct grids spent 70.4 s of its 81.2 s invoke in
        # synth (11.7 s/image) versus 0.15 s/image on a job with 2 grids. Warming two shapes is
        # not enough; the real grids are all different.
        if _COMPILE:
            net.dose = torch.compile(net.dose, dynamic=True)
        with torch.no_grad(), torch.autocast("cuda"):
            for s in (32, 48):
                net.dose(torch.zeros(1, 5, 32, s, s, device=DEV),
                         torch.zeros(1, dtype=torch.long, device=DEV))
            for zyx in ((128, 160, 160), (144, 176, 176)):
                clf(torch.zeros(1, 1, *zyx, device=DEV))    # clf is eager; just warms cuDNN
        torch.cuda.synchronize()
    _STATE.update(machine=machine, pm=pm, net=net, clf=clf, ready=True,
                  density_direct=bool(cfg.get("density_direct", True)),
                  native_grid=bool(cfg.get("native_synth", False)))
    print("[app] proton-MRI models loaded + compiled; ready", flush=True)


class _Img:
    def __init__(self, sitk_img, density_np):
        import SimpleITK as sitk
        self.array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        self.spacing = tuple(sitk_img.GetSpacing()); self.origin = tuple(sitk_img.GetOrigin())


def _output_info_by_key(beams):
    return {(b.get("beam_idx", bi), ri, l): bl["output_info"]
            for bi, b in enumerate(beams)
            for ri, r in enumerate(b["rays"])
            for l, bl in enumerate(r["beamlets"])}


def _predict_fn(sitk_img, entry, on_frame=None):
    """With `on_frame`, each beamlet goes to gc_invoke the moment it is computed, so the (slow,
    network-mounted) /output write overlaps the GPU work instead of following it."""
    machine, pm, net, clf = _STATE["machine"], _STATE["pm"], _STATE["net"], _STATE["clf"]
    # density_direct from config: proton-native synth outputs density (sct01*2.5); a HU-space synth
    # (e.g. the photon scheme2-p4 sCT dropped in) needs density_direct=False -> hu_to_density(hu_anchors).
    density_np, _ = synth_density(sitk_img, clf, net, DEV,
                                  density_direct=_STATE.get("density_direct", True),
                                  hu_anchors=machine.hu_anchors,
                                  timings=_STATE["timings"],
                                  native_grid=_STATE.get("native_grid", False))
    img = _Img(sitk_img, density_np)
    density_t = torch.as_tensor(density_np, device=DEV)
    beams = entry["beams"]
    oinfo = _output_info_by_key(beams)

    if on_frame is not None:
        sink = lambda key, dose, bbox: (on_frame(dose, bbox, oinfo[key]) if key in oinfo else None)
        predict_beams(img, beams, density_np, density_t, net.dose, pm, DEV, on_frame=sink)
        return {}

    preds = predict_beams(img, beams, density_np, density_t, net.dose, pm, DEV)
    return {k: (dose, bbox, oinfo[k]) for k, (dose, bbox) in preds.items() if k in oinfo}


def run_invoke():
    t0 = time.time()
    st = {}
    # the MRI path is compute-bound (photon-MRI 103s vs its byte-identical photon-CT
    # twin at 46s), and all of the extra is sCT synthesis — split it per stage
    _STATE["timings"] = {}
    slots = gc_invoke.process_run(INPUT, OUTPUT, _predict_fn, "proton", "mri", stats=st)
    n = sum(len(v) for v in slots.values())
    dt = time.time() - t0
    # B (dosemap count) is the regressor the evaluator fits dosemap_time against, and dosemap_time
    # is both the ranked efficiency metric and what the 1 s/beam gate applies to. The compute/write
    # split matters because on the platform /output is a slow network mount, not local disk.
    gb = st.get("out_gb", 0.0)
    if st.get("overlapped"):
        # compute_s = producer wall time; the rest of the invoke is the writers draining. It is the
        # FLOOR any write-side optimisation can reach, so it decides whether more write work pays.
        # write_busy is summed over streams: busy/invoke ~1 with pool>1 means the storage is not
        # scaling with concurrency; ~pool means every stream stayed busy.
        wb, pool, cs = st.get("write_busy_s", 0.0), st.get("write_pool", 1), st.get("compute_s", 0.0)
        split = (f"compute {cs:.1f}s + drain {dt-cs:.1f}s | write {wb:.1f}s agg over pool {pool} "
                 f"= {wb/max(dt,1e-9):.2f}x invoke, {gb*1024/max(wb,1e-9):.0f} MB/s per stream")
        e, l = st.get("early_ms"), st.get("late_ms")
        warm = (f" | first frame {st.get('first_frame_s',0):.1f}s, "
                f"gap first10 {e:.0f}ms vs rest {l:.0f}ms" if e and l else "")
        extra = (f" | backlog<={st.get('backlog',0)} | peak RSS {st.get('peak_rss_gb',0):.2f} GB"
                 f" | /output = {st.get('out_fs','?')}{warm}")
    else:
        split = f"compute {st.get('compute_s',0):.1f}s + write {st.get('write_s',0):.1f}s"
        extra = ""
    tm = _STATE.get("timings", {})
    extra += (" | sCT " + " ".join(f"{k} {v:.1f}s" for k, v in sorted(tm.items()))
              + f" (total {sum(tm.values()):.1f}s)") if tm else ""
    print(f"[app] invoke done in {dt:.1f}s | {n} dosemaps | {dt/max(n,1)*1000:.0f} ms/dosemap "
          f"| {split} | out {gb:.2f} GB ({gb*1024/max(dt,1e-9):.0f} MB/s) "
          f"| missing {st.get('missing',0)}{extra} | grids {st.get('grids')}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if (self.path == "/health" and _STATE["ready"]) else
                           (503 if self.path == "/health" else 404)); self.end_headers()

    def do_POST(self):
        if self.path == "/invoke":
            try:
                run_invoke(); self.send_response(201); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                import traceback; traceback.print_exc()
                self.send_response(500); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    load_models()
    ThreadingHTTPServer(("0.0.0.0", 4743), Handler).serve_forever()
