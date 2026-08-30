"""Fine-grained profile of the PHOTON deploy per-CP path (container/photon/predict.predict_cps) to
find the bottleneck. Split: photon_channels(BEV fan) | normalize | pad | net(synced) | .cpu xfer.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_profile_photon_deploy.py 1ABB006 80
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.channels import photon_channels
from doserad.inference.pipeline import _normalize_gpu, _build_coords
from doserad.eval.plan_predict import pad_to_multiple
from doserad.data.dataset import DOSE_SCALE
from doserad.io.mha import load_mha

DEV = "cuda"
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
W = "/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt"
MACH = f"{ROOT}/beam_parameters.json"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 80


def main():
    machine = load_photon_machine(MACH)
    net = DoseUNet3D(in_ch=6, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load(W, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    net = torch.compile(net, dynamic=True)
    mod = torch.zeros(1, dtype=torch.long, device=DEV)

    img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    density_np = hu_to_density(img.array, machine.hu_anchors)
    coords = _build_coords(img, DEV)
    plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))

    T = {k: 0.0 for k in ["channels", "norm", "pad", "net", "xfer"]}
    n = 0; warmed = False
    for bi, b in enumerate(plan["beams"]):
        iso = b.get("iso_center", [0, 0, 0])
        for ci, cp in enumerate(b["control_points"]):
            torch.cuda.synchronize(); t = time.perf_counter()
            crop, bbox = photon_channels(image=img, machine=machine, iso_xyz=iso, gantry_deg=cp["gantry_angle"],
                                         mlc_left=np.asarray(cp["mlc_left_int_mm"]), mlc_right=np.asarray(cp["mlc_right_int_mm"]),
                                         density_override=density_np, coords=coords, crop_margin=8, return_tensor=True)
            torch.cuda.synchronize(); T["channels"] += time.perf_counter() - t
            t = time.perf_counter()
            crop = _normalize_gpu(crop, True)
            torch.cuda.synchronize(); T["norm"] += time.perf_counter() - t
            t = time.perf_counter()
            x = crop.unsqueeze(0); x_pad, orig = pad_to_multiple(x, factor=8)
            torch.cuda.synchronize(); T["pad"] += time.perf_counter() - t
            t = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda"):
                y = net(x_pad, mod)
            torch.cuda.synchronize(); T["net"] += time.perf_counter() - t
            t = time.perf_counter()
            d, h, w = orig
            dose = (y[0, 0, :d, :h, :w].float() / DOSE_SCALE).cpu().numpy().astype(np.float32)
            T["xfer"] += time.perf_counter() - t
            n += 1
            if not warmed and n >= 8:
                for k in T: T[k] = 0.0
                n = 0; warmed = True
            if n >= CAP:
                break
        if n >= CAP:
            break
    tot = sum(T.values())
    print(f"\n=== PHOTON DEPLOY PROFILE ({PID}, {n} warm CPs) ===")
    for k in ["channels", "norm", "pad", "net", "xfer"]:
        print(f"  {k:>9}: {T[k]/max(n,1)*1000:>7.2f} ms/CP  ({T[k]/tot*100:>4.1f}%)")
    print(f"  {'TOTAL':>9}: {tot/max(n,1)*1000:>7.2f} ms/CP")


if __name__ == "__main__":
    main()
