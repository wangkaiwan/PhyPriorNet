"""Grand-Challenge invoke server for the proton dose-calculation algorithm.

Lifecycle (submission_instructions_2026-07-17.md §3):
  - startup: load the dose net + machine, torch.compile(dynamic) warm on a dummy shape (all during
    the FREE pre-/health time), then serve.
  - GET  /health  -> 200 once ready.
  - POST /invoke  -> process /input, write /output, 201.
No network at invoke; only /output + scratch are writable.

Env: DOSERAD_MODALITY (ct|mri, default ct), DOSERAD_WEIGHTS, DOSERAD_MACHINE, DOSERAD_INPUT,
DOSERAD_OUTPUT, TORCHINDUCTOR_CACHE_DIR.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from container.proton import gc_invoke
from container.proton.predict import predict_beams

MODALITY = os.environ.get("DOSERAD_MODALITY", "ct")
WEIGHTS = os.environ.get("DOSERAD_WEIGHTS", "/opt/algorithm/weights/proton.pt")
MACHINE = os.environ.get("DOSERAD_MACHINE", "/opt/algorithm/weights/beam_parameters.json")
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
_STATE = {"ready": False}


def load_models():
    machine = load_photon_machine(MACHINE)
    pm = ProtonMachineData(device=DEV)
    net = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load(WEIGHTS, map_location=DEV)
    net.load_state_dict(sd.get("ema", sd.get("model")))
    cnet = torch.compile(net, dynamic=True) if (DEV == "cuda" and _COMPILE) else net
    # warm the dynamic-compiled kernels on a dummy shape (FREE, pre-/health)
    if DEV == "cuda":
        with torch.no_grad(), torch.autocast("cuda"):
            for s in (32, 48):
                cnet(torch.zeros(1, 5, 32, s, s, device=DEV), torch.zeros(1, dtype=torch.long, device=DEV))
        torch.cuda.synchronize()
    _STATE.update(machine=machine, pm=pm, net=cnet, ready=True)
    print("[app] models loaded + compiled; ready", flush=True)


class _Img:
    def __init__(self, sitk_img):
        import SimpleITK as sitk
        self.array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)   # (z,y,x)
        self.spacing = tuple(sitk_img.GetSpacing())                        # (sx,sy,sz)
        self.origin = tuple(sitk_img.GetOrigin())


def _output_info_by_key(beams):
    return {(b.get("beam_idx", bi), ri, l): bl["output_info"]
            for bi, b in enumerate(beams)
            for ri, r in enumerate(b["rays"])
            for l, bl in enumerate(r["beamlets"])}


def _predict_fn(sitk_img, entry, on_frame=None):
    """gc_invoke predict_fn: sitk image + metadata entry -> {beam_key: (dose_np, bbox, output_info)}.
    With `on_frame` it instead streams each beamlet to gc_invoke as soon as it is computed, so the
    (slow, network-mounted) /output write overlaps the GPU work instead of following it."""
    img = _Img(sitk_img)
    machine, pm, net = _STATE["machine"], _STATE["pm"], _STATE["net"]
    density_np = hu_to_density(img.array, machine.hu_anchors)
    density_t = torch.as_tensor(density_np, device=DEV)
    beams = entry["beams"]
    oinfo = _output_info_by_key(beams)

    if on_frame is not None:
        sink = lambda key, dose, bbox: (on_frame(dose, bbox, oinfo[key]) if key in oinfo else None)
        predict_beams(img, beams, density_np, density_t, net, pm, DEV, on_frame=sink)
        return {}

    preds = predict_beams(img, beams, density_np, density_t, net, pm, DEV)
    # gc_invoke wants {key: (crop_np, bbox, output_info)}; it lazily places the small crop into a
    # full-grid frame at write time (memory: only one full frame, not n).
    return {k: (dose, bbox, oinfo[k]) for k, (dose, bbox) in preds.items() if k in oinfo}


def run_invoke():
    t0 = time.time()
    st = {}
    slots = gc_invoke.process_run(INPUT, OUTPUT, _predict_fn, "proton", MODALITY, stats=st)
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
    print(f"[app] invoke done in {dt:.1f}s | {n} dosemaps | {dt/max(n,1)*1000:.0f} ms/dosemap "
          f"| {split} | out {gb:.2f} GB ({gb*1024/max(dt,1e-9):.0f} MB/s) "
          f"| missing {st.get('missing',0)}{extra} | grids {st.get('grids')}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200 if _STATE["ready"] else 503); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/invoke":
            try:
                run_invoke()
                self.send_response(201); self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                import traceback; traceback.print_exc()
                self.send_response(500); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    load_models()
    ThreadingHTTPServer(("0.0.0.0", 4743), Handler).serve_forever()
