"""Clean end-to-end CT deploy timing (full plan, one patient, cold-inclusive) — SAME methodology as
the MRI container gamma tests, so all 4 tasks are comparable. Accel toggled by env
(DOSERAD_2PASS for proton, DOSERAD_PHOTON_FAST for photon).
  CUDA_VISIBLE_DEVICES=1 DOSERAD_2PASS=1 conda run -n doserad python accel/exp_ct_endtoend.py proton 1ABB006
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.io.mha import load_mha

DEV = "cuda"
PARTICLE = sys.argv[1] if len(sys.argv) > 1 else "proton"
PID = sys.argv[2] if len(sys.argv) > 2 else "1ABB006"
MACH = "/data/kwang/DoseRad2026_raw/beam_parameters.json"


def main():
    machine = load_photon_machine(MACH)
    if PARTICLE == "proton":
        from container.proton.predict import predict_beams, _USE_2PASS
        ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
        pm = ProtonMachineData(device=DEV)
        net = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
        sd = torch.load("/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt", map_location=DEV)
        net.load_state_dict(sd.get("ema", sd.get("model"))); net = torch.compile(net, dynamic=True)
        img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
        dn = hu_to_density(img.array, machine.hu_anchors); dt = torch.as_tensor(dn, device=DEV)
        plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
        for bi, b in enumerate(plan["beams"]):
            b["beam_idx"] = bi

        class _I:
            array = img.array; spacing = img.spacing; origin = img.origin
        t0 = time.time()
        preds = predict_beams(_I(), plan["beams"], dn, dt, net, pm, DEV)
        dt_s = time.time() - t0
        nb = len(preds)
        print(f"PROTON-CT ({PID}) 2pass={_USE_2PASS}: {nb} beamlets, {dt_s:.1f}s -> {dt_s/nb*1000:.1f} ms/beamlet")
    else:
        from container.photon.predict import predict_cps, _USE_FAST
        ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
        net = DoseUNet3D(in_ch=6, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
        sd = torch.load("/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt", map_location=DEV)
        net.load_state_dict(sd.get("ema", sd.get("model"))); net = torch.compile(net, dynamic=True)
        img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
        dn = hu_to_density(img.array, machine.hu_anchors)
        plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
        for bi, b in enumerate(plan["beams"]):
            b["beam_idx"] = bi
        t0 = time.time()
        preds = predict_cps(img, plan["beams"], dn, net, machine, DEV)
        dt_s = time.time() - t0
        nc = len(preds)
        print(f"PHOTON-CT ({PID}) fast={_USE_FAST}: {nc} CPs, {dt_s:.1f}s -> {dt_s/nc*1000:.1f} ms/CP")


if __name__ == "__main__":
    main()
