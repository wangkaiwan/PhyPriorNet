"""Split the photon channels 50ms/CP: how much is the FULL-VOLUME aperture projection (open_mask
over coords_full) vs radiological_depth_fast? Decides whether an analytic-aperture-bbox 2-pass
(compute channels cropped) is worth building. Times, per CP: full-vol projection block | rdepth.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_photon_chan_split.py 1ABB006 60
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.channels import _mask_bbox
from doserad.physics.raytrace import radiological_depth_fast
from doserad.inference.pipeline import _build_coords
from doserad.io.mha import load_mha

DEV = "cuda"
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
MACH = f"{ROOT}/beam_parameters.json"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def main():
    machine = load_photon_machine(MACH)
    img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    density = hu_to_density(img.array, machine.hu_anchors)
    nz, ny, nx = img.array.shape
    coords_full = _build_coords(img, DEV)   # (z,y,x,3)
    plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    T = {"proj": 0.0, "rdepth": 0.0}; n = 0; warmed = False
    for b in plan["beams"]:
        iso = np.asarray(b.get("iso_center", [0, 0, 0]), np.float64); iso_t = torch.as_tensor(iso, dtype=torch.float32, device=DEV)
        for cp in b["control_points"]:
            g = cp["gantry_angle"]
            src_np = beam_source_pos(iso, machine.sad_mm, g); axis_np, u_np, v_np = beam_basis(g)
            src = torch.as_tensor(src_np, dtype=torch.float32, device=DEV)
            axis = torch.as_tensor(axis_np, dtype=torch.float32, device=DEV)
            u_hat = torch.as_tensor(u_np, dtype=torch.float32, device=DEV); v_hat = torch.as_tensor(v_np, dtype=torch.float32, device=DEV)
            ml = torch.as_tensor(np.asarray(cp["mlc_left_int_mm"]), dtype=torch.float32, device=DEV)
            mr = torch.as_tensor(np.asarray(cp["mlc_right_int_mm"]), dtype=torch.float32, device=DEV)
            torch.cuda.synchronize(); t = time.perf_counter()
            vec = coords_full - src; denom = (vec * axis).sum(-1)
            t_iso = ((iso_t - src) * axis).sum() / torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
            hit = src + t_iso.unsqueeze(-1) * vec; rel = hit - iso_t
            uu = (rel * u_hat).sum(-1); vv = (rel * v_hat).sum(-1)
            half = machine.num_leaf_pairs / 2.0
            pair = torch.floor(vv / machine.leaf_thickness_mm + half).long()
            valid = (pair >= 0) & (pair < machine.num_leaf_pairs); pidx = pair.clamp(0, machine.num_leaf_pairs - 1)
            jx, jy = machine.jaw_x_mm, machine.jaw_y_mm
            om = (valid & (ml[pidx] < mr[pidx]) & (uu >= ml[pidx]) & (uu <= mr[pidx]) &
                  (uu >= jx[0]) & (uu <= jx[1]) & (vv >= jy[0]) & (vv <= jy[1])).float()
            bbox = _mask_bbox(om, 8, (nz, ny, nx))
            torch.cuda.synchronize(); T["proj"] += time.perf_counter() - t
            t = time.perf_counter()
            _ = radiological_depth_fast(density, img.spacing, img.origin, src_np, axis_np, u_np, v_np, iso,
                                        out_bbox=bbox, coords=coords_full)
            torch.cuda.synchronize(); T["rdepth"] += time.perf_counter() - t
            n += 1
            if not warmed and n >= 8:
                for k in T: T[k] = 0.0
                n = 0; warmed = True
            if n >= CAP:
                break
        if n >= CAP:
            break
    tot = sum(T.values())
    print(f"\n=== PHOTON CHANNELS SPLIT ({PID}, {n} warm CPs) ===")
    for k in ["proj", "rdepth"]:
        print(f"  {k:>8}: {T[k]/max(n,1)*1000:>7.2f} ms/CP  ({T[k]/tot*100:>4.1f}%)")
    print(f"  full-vol projection is the aperture open_mask over ALL {nz*ny*nx/1e6:.1f}M voxels")


if __name__ == "__main__":
    main()
