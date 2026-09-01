"""Aggregate check: over many beamlets, is the skin-entry PB prior at least as accurate as
the current (from-source+offset) prior vs MC?  Metrics per beamlet: (1) Bragg-peak position
error along the beam axis (mm), (2) normalized dose MAE over MC>=10% voxels. CPU by default.
"""
import os, sys, json, glob
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData, proton_pb_dose_gpu
from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"; CACHE = "/data/kwang/doserad_cache_archive/proton"
DEV = os.environ.get("DEV", "cpu")


def axis_peak(vol, bbox, src, tgt, org, sp):
    ox, oy, oz = org; sx, sy, sz = sp; z0, z1, y0, y1, x0, x1 = bbox
    axis = np.asarray(tgt, float) - np.asarray(src, float); axis /= np.linalg.norm(axis)
    pk = np.unravel_index(int(np.argmax(vol)), vol.shape)
    pw = np.array([ox + (x0 + pk[2]) * sx, oy + (y0 + pk[1]) * sy, oz + (z0 + pk[0]) * sz])
    tt = np.arange(-60, 60, 0.5); w = pw[None] + tt[:, None] * axis[None]
    nz, ny, nx = vol.shape
    iz = np.clip(((w[:, 2] - oz) / sz - z0).round().astype(int), 0, nz - 1)
    iy = np.clip(((w[:, 1] - oy) / sy - y0).round().astype(int), 0, ny - 1)
    ix = np.clip(((w[:, 0] - ox) / sx - x0).round().astype(int), 0, nx - 1)
    return tt, vol[iz, iy, ix]


def norm_mae(a, mc):
    m = mc >= 0.1 * mc.max()
    s = mc[m].max() + 1e-9
    return float(np.abs(a[m] - mc[m]).mean() / s)


def main(pids, nper=20):
    hu_anchors = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json").hu_anchors
    pm = ProtonMachineData(device=DEV)
    dpk = {"current": [], "mask": [], "skin-entry": []}; mae = {"current": [], "skin-entry": []}
    for pid in pids:
        ct = load_mha(f"{ROOT}/{pid}/image/ct.mha"); plan = json.load(open(f"{ROOT}/{pid}/{pid}.json"))
        dens = hu_to_density(ct.array, hu_anchors).astype(np.float32)
        rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]): (r["ray_source"], r["ray_target"], bl["energy"])
                for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
        files = sorted(glob.glob(f"{CACHE}/{pid}/B*_R*_L*.npz"), key=lambda f: -float(np.load(f)["dose_max"]))[:nper]
        for f in files:
            name = os.path.basename(f)[:-4]; b, r, l = [int(x[1:]) for x in name.split("_")]
            npz = np.load(f); bbox = tuple(npz["bbox"]); mc = npz["dose"].astype(np.float32)
            src, tgt, e = rays[(b, r, l)]
            kw = dict(out_bbox=bbox, machine=pm, density_override=dens, device=DEV)
            cur = proton_pb_dose_gpu(ct, src, tgt, e, **kw)
            skn = proton_pb_dose_gpu_skinentry(ct, src, tgt, e, out_bbox=bbox, machine=pm, density_override=dens, device=DEV)
            tt, pmc = axis_peak(mc, bbox, src, tgt, ct.origin, ct.spacing)
            _, pcu = axis_peak(cur, bbox, src, tgt, ct.origin, ct.spacing)
            _, psk = axis_peak(skn, bbox, src, tgt, ct.origin, ct.spacing)
            mcpk = tt[int(np.argmax(pmc))]
            dpk["current"].append(tt[int(np.argmax(pcu))] - mcpk)
            dpk["skin-entry"].append(tt[int(np.argmax(psk))] - mcpk)
            mae["current"].append(norm_mae(cur, mc)); mae["skin-entry"].append(norm_mae(skn, mc))
        print(f"  {pid}: {len(files)} beamlets done", flush=True)
    print("\n=== AGGREGATE (n=%d beamlets) ===" % len(dpk["current"]))
    for k in ["current", "skin-entry"]:
        a = np.abs(dpk[k]); print(f"  {k:11s}: |peak err| mean {a.mean():.2f}mm (signed {np.mean(dpk[k]):+.2f}), "
                                  f"dose MAE {np.mean(mae[k])*100:.2f}%")
    print(f"  -> skin-entry {'BETTER' if np.abs(dpk['skin-entry']).mean() < np.abs(dpk['current']).mean() else 'not better'} "
          f"on peak; {'BETTER' if np.mean(mae['skin-entry']) < np.mean(mae['current']) else 'not better'} on dose MAE")


if __name__ == "__main__":
    pids = sys.argv[1].split(",") if len(sys.argv) > 1 else ["1ABB006", "1ABC001", "1PBA003"]
    main(pids, int(sys.argv[2]) if len(sys.argv) > 2 else 20)
