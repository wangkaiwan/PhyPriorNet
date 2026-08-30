"""Grand-Challenge invoke server for the PHOTON dose-calculation algorithm. Mirrors the proton
app; photon bbox comes from the MLC aperture (photon_channels crop_margin=8) so no GT is needed.
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
from container.proton import gc_invoke
from container.photon.predict import predict_cps

MODALITY = os.environ.get("DOSERAD_MODALITY", "ct")
WEIGHTS = os.environ.get("DOSERAD_WEIGHTS", "/opt/algorithm/weights/photon.pt")
MACHINE = os.environ.get("DOSERAD_MACHINE", "/opt/algorithm/weights/beam_parameters.json")
# second checkpoint for thoracic cases; unset = routing off, single model as before
WEIGHTS_THORACIC = os.environ.get("DOSERAD_WEIGHTS_THORACIC", "")
INPUT = Path(os.environ.get("DOSERAD_INPUT", "/input"))
OUTPUT = Path(os.environ.get("DOSERAD_OUTPUT", "/output"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_STATE = {"ready": False}


def _weights_of(path):
    sd = torch.load(path, map_location=DEV)
    sd = sd.get("ema", sd.get("model", sd))
    if any(k.startswith("dose.") for k in sd):        # an E2E checkpoint (synth.* + dose.*)
        sd = {k[len("dose."):]: v for k, v in sd.items() if k.startswith("dose.")}
    return sd


def load_models():
    """Optionally route by anatomical site, which the organizers confirmed is allowed.

    Two checkpoints have complementary failure modes: the dedicated photon-CT net is better on
    thoracic cases while the multimodal one avoids catastrophic abdominal failures. Scored on the
    corrected metric they are 93.26 vs 88.50 on lung and 91.42 vs 94.02 on abdomen, so picking per
    site beats either alone.

    They share an architecture, so ONE compiled module serves both and the second checkpoint is
    just a state_dict. That matters: photon keeps torch.compile (~3x, and it is compute-bound), a
    second compile would double the ~100 s startup, and photon's total already sits near
    SageMaker's MaxRuntimeInSeconds. load_state_dict copies IN PLACE so parameter tensors keep
    their identity and the inductor graph stays valid -- verified: swapping costs 14 ms, changes
    the output, round-trips exactly, and both weight sets match their eager counterparts to 1.2e-3.
    """
    machine = load_photon_machine(MACHINE)
    sds = {"default": _weights_of(WEIGHTS)}
    # infer width from the checkpoint instead of hardcoding 48, so a distilled student (base 32)
    # loads without an app change
    base = int(sds["default"]["stem.weight"].shape[0])
    net = DoseUNet3D(in_ch=6, base=base, levels=4, bottleneck="dilated").to(DEV).eval()
    if WEIGHTS_THORACIC:
        sds["thoracic"] = _weights_of(WEIGHTS_THORACIC)
    net.load_state_dict(sds["default"])
    # speed knobs (both default OFF -> byte-identical): DOSERAD_HALF=1 casts weights to fp16 (autocast
    # already computes fp16; this removes weight-cast traffic). DOSERAD_COMPILE_MODE=max-autotune uses
    # the slower-startup, faster-kernel inductor mode (photon startup is free — platform-proven).
    if os.environ.get("DOSERAD_HALF", "0") == "1":
        net = net.half()
    _cmode = os.environ.get("DOSERAD_COMPILE_MODE", "")
    if DEV == "cuda":
        cnet = torch.compile(net, dynamic=True, mode=_cmode) if _cmode else torch.compile(net, dynamic=True)
    else:
        cnet = net
    if DEV == "cuda":
        with torch.no_grad(), torch.autocast("cuda"):
            for s in (64, 96):
                cnet(torch.zeros(1, 6, 64, s, s, device=DEV), torch.zeros(1, dtype=torch.long, device=DEV))
        torch.cuda.synchronize()
    _STATE.update(machine=machine, net=cnet, raw=net, sds=sds, active="default", ready=True)
    print(f"[app] photon models loaded + compiled; ready "
          f"(routing: {'thoracic->2nd model' if WEIGHTS_THORACIC else 'off'})", flush=True)


def _select(region):
    """Swap in the site's weights if they differ from what is currently loaded (14 ms, no recompile)."""
    want = "thoracic" if (region or "").lower().startswith("thorac") and "thoracic" in _STATE["sds"] \
        else "default"
    if want != _STATE["active"]:
        _STATE["raw"].load_state_dict(_STATE["sds"][want])
        _STATE["active"] = want
    return want


class _Img:
    def __init__(self, sitk_img):
        import SimpleITK as sitk
        self.array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        self.spacing = tuple(sitk_img.GetSpacing())
        self.origin = tuple(sitk_img.GetOrigin())


# --- low-resolution inference (DOSERAD_PHOTON_SPACING, e.g. "3") ------------------------------
# Predict on a coarser grid (net forward ~= 85% of runtime, scales with voxels: 3mm = 0.30x of
# 2mm), then upsample each CP dose crop back to the SOURCE grid so gc_invoke and every metric see
# the contracted grid. Default (env unset) leaves the pipeline byte-identical.
_LOWRES = float(os.environ.get("DOSERAD_PHOTON_SPACING", "0") or 0)


def _resample_iso(sitk_img, sp, default):
    import SimpleITK as sitk
    sz, osp = sitk_img.GetSize(), sitk_img.GetSpacing()
    nsz = [int(round(sz[i] * osp[i] / sp)) for i in range(3)]
    return sitk.Resample(sitk_img, nsz, sitk.Transform(), sitk.sitkLinear, sitk_img.GetOrigin(),
                         (sp, sp, sp), sitk_img.GetDirection(), default, sitk.sitkFloat32)


def _upsample_crop(dose_np, bbox_lo, img_src, sp_lo):
    """Linear-resample a low-res crop (z,y,x @ sp_lo, same origin/direction as source) onto the
    source-grid voxels it covers. Returns (dose_src_np, bbox_src). World: src j -> lo k = j*s/sp - lo0."""
    z0, z1, y0, y1, x0, x1 = (int(v) for v in bbox_lo)
    ssz, ssy, ssx = img_src.spacing[2], img_src.spacing[1], img_src.spacing[0]
    nz, ny, nx = img_src.array.shape
    j = lambda lo, s: int(np.ceil(lo * sp_lo / s - 1e-6))
    J = lambda hi, s, n: min(int(np.floor(hi * sp_lo / s + 1e-6)), n - 1)
    jz0, jz1 = j(z0, ssz), J(z1, ssz, nz)
    jy0, jy1 = j(y0, ssy), J(y1, ssy, ny)
    jx0, jx1 = j(x0, ssx), J(x1, ssx, nx)
    if jz1 < jz0 or jy1 < jy0 or jx1 < jx0:
        return None, None
    t = torch.as_tensor(dose_np, dtype=torch.float32, device=DEV)[None, None]
    dz, dy, dx = t.shape[-3:]
    kz = (torch.arange(jz0, jz1 + 1, device=DEV, dtype=torch.float32) * ssz / sp_lo) - z0
    ky = (torch.arange(jy0, jy1 + 1, device=DEV, dtype=torch.float32) * ssy / sp_lo) - y0
    kx = (torch.arange(jx0, jx1 + 1, device=DEV, dtype=torch.float32) * ssx / sp_lo) - x0
    # normalized (align_corners=True): -1..1 maps to voxel centers 0..N-1
    gz = (2 * kz / max(dz - 1, 1)) - 1
    gy = (2 * ky / max(dy - 1, 1)) - 1
    gx = (2 * kx / max(dx - 1, 1)) - 1
    grid = torch.stack(torch.meshgrid(gz, gy, gx, indexing="ij"), dim=-1)[None]   # (1,Z,Y,X,3) as z,y,x
    grid = grid[..., [2, 1, 0]]                                                    # grid_sample wants x,y,z
    out = torch.nn.functional.grid_sample(t, grid, mode="bilinear", align_corners=True)
    return out[0, 0].cpu().numpy().astype(np.float32), (jz0, jz1, jy0, jy1, jx0, jx1)


def _output_info_by_key(beams):
    return {(b.get("beam_idx", bi), ci): cp["output_info"]
            for bi, b in enumerate(beams)
            for ci, cp in enumerate(b["control_points"])}


def _predict_fn(sitk_img, entry, on_frame=None):
    """With `on_frame`, each control point goes to gc_invoke the moment it is computed, so the
    (slow, network-mounted) /output write overlaps the GPU work instead of following it."""
    img_src = _Img(sitk_img)
    _select(entry.get("anatomical_region"))        # platform values: "thoracic" | "abdominal"
    machine, net = _STATE["machine"], _STATE["net"]
    beams = entry["beams"]
    oinfo = _output_info_by_key(beams)

    if _LOWRES > 0:
        img = _Img(_resample_iso(sitk_img, _LOWRES, -1000.0))
        density_np = hu_to_density(img.array, machine.hu_anchors)
        if on_frame is not None:
            def sink(key, dose, bbox):
                if key not in oinfo:
                    return
                d2, b2 = _upsample_crop(dose, bbox, img_src, _LOWRES)
                if d2 is not None:
                    on_frame(d2, b2, oinfo[key])
            predict_cps(img, beams, density_np, net, machine, DEV, on_frame=sink)
            return {}
        preds = predict_cps(img, beams, density_np, net, machine, DEV)
        out = {}
        for k, (dose, bbox) in preds.items():
            if k not in oinfo:
                continue
            d2, b2 = _upsample_crop(dose, bbox, img_src, _LOWRES)
            if d2 is not None:
                out[k] = (d2, b2, oinfo[k])
        return out

    img = img_src
    density_np = hu_to_density(img.array, machine.hu_anchors)
    if on_frame is not None:
        sink = lambda key, dose, bbox: (on_frame(dose, bbox, oinfo[key]) if key in oinfo else None)
        predict_cps(img, beams, density_np, net, machine, DEV, on_frame=sink)
        return {}

    preds = predict_cps(img, beams, density_np, net, machine, DEV)
    return {k: (dose, bbox, oinfo[k]) for k, (dose, bbox) in preds.items() if k in oinfo}


def run_invoke():
    t0 = time.time()
    st = {}
    slots = gc_invoke.process_run(INPUT, OUTPUT, _predict_fn, "photon", MODALITY, stats=st)
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
