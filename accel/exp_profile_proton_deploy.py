"""Fine-grained profile of the proton DEPLOY per-beamlet path (container/proton/predict) to find the
real bottleneck. Docs lumped 'tight+net ~63ms' but the net forward is only ~2-5ms — so most of that
is overhead. Split it: geom_bbox | build_ray | tight_from_pb | crop+pad | net(synced) | .cpu transfer.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_profile_proton_deploy.py 1ABB006 60
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from doserad.io.mha import load_mha
from accel.proton_build_ray import build_ray
from accel.proton_2pass import build_ray_2pass
from container.proton.geom_bbox import geom_bbox_proton
from container.proton.predict import _tight_from_pb

_2PASS = os.environ.get("DOSERAD_2PASS", "1") != "0"

DEV = "cuda"
ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
W = "/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt"
MACH = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
_r16 = lambda v: -(-v // 16) * 16


def main():
    machine = load_photon_machine(MACH); pm = ProtonMachineData(device=DEV)
    net = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load(W, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    net = torch.compile(net, dynamic=True)
    mod = torch.zeros(1, dtype=torch.long, device=DEV)

    img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    density_np = hu_to_density(img.array, machine.hu_anchors)
    density_t = torch.as_tensor(density_np, device=DEV)
    plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi

    T = {k: 0.0 for k in ["geom", "build", "tight", "croppad", "net", "xfer"]}
    n = 0
    warmed = False
    for b in plan["beams"]:
        for ri, r in enumerate(b["rays"]):
            bls = []
            t = time.perf_counter()
            for l, bl in enumerate(r["beamlets"]):
                if n + len(bls) >= CAP:
                    break
                gb = geom_bbox_proton(density_np, img.spacing, img.origin, r["ray_source"],
                                      r["ray_target"], pm, bl["energy"])
                if gb is not None:
                    bls.append(dict(energy=bl["energy"], bbox=gb, key=(b["beam_idx"], ri, l)))
            T["geom"] += time.perf_counter() - t
            if not bls:
                continue
            torch.cuda.synchronize(); t = time.perf_counter()
            _bfn = build_ray_2pass if _2PASS else build_ray
            stacks = _bfn(img, r["ray_source"], r["ray_target"], bls, machine=pm, density=density_t, device=DEV)
            torch.cuda.synchronize(); T["build"] += time.perf_counter() - t
            for (stack, gbb), bl in zip(stacks, bls):
                t = time.perf_counter()
                pb = stack[2] * _P_CH_SCALE_PRIOR[2] / PROTON_DOSE_SCALE
                tb = _tight_from_pb(pb)
                torch.cuda.synchronize(); T["tight"] += time.perf_counter() - t
                if tb is None:
                    continue
                z0, z1, y0, y1, x0, x1 = tb
                t = time.perf_counter()
                tight = stack[:, z0:z1+1, y0:y1+1, x0:x1+1]
                d, h, w = tight.shape[-3:]
                x = F.pad(tight[None], (0, _r16(w)-w, 0, _r16(h)-h, 0, _r16(d)-d))
                torch.cuda.synchronize(); T["croppad"] += time.perf_counter() - t
                t = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda"):
                    y = net(x, mod)
                torch.cuda.synchronize(); T["net"] += time.perf_counter() - t
                t = time.perf_counter()
                dose = (y[0, 0, :d, :h, :w].float() / PROTON_DOSE_SCALE).cpu().numpy().astype(np.float32)
                T["xfer"] += time.perf_counter() - t
                n += 1
                if not warmed and n >= 8:   # discard warmup: reset counters after 8 beamlets
                    for k in T: T[k] = 0.0
                    n = 0; warmed = True
            if n >= CAP:
                break
        if n >= CAP:
            break

    tot = sum(T.values())
    print(f"\n=== PROTON DEPLOY PROFILE ({PID}, {n} warm beamlets) ===")
    for k in ["geom", "build", "tight", "croppad", "net", "xfer"]:
        print(f"  {k:>8}: {T[k]/max(n,1)*1000:>7.2f} ms/beamlet  ({T[k]/tot*100:>4.1f}%)")
    print(f"  {'TOTAL':>8}: {tot/max(n,1)*1000:>7.2f} ms/beamlet")


if __name__ == "__main__":
    main()
