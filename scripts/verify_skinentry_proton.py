"""Verify the per-ray proton skin-entry PB prior against MC on a single beamlet:
compare current (from-source + rad_depth_offset), mask_air, and skin-entry priors to the
cached MC dose. Renders depth-dose profile along the beamlet axis + 2D slices.
Runs on CPU by default (so it does not contend with training GPUs).  MPLBACKEND=Agg.
"""
import os, sys, json, glob
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData, proton_pb_dose_gpu
from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
CACHE = "/data/kwang/doserad_cache_archive/proton"
DEV = os.environ.get("DEV", "cpu")


def trilerp(vol, pts_ijk):  # vol (z,y,x); pts (...,3) in (i=z,j=y,k=x) float idx
    z, y, x = [pts_ijk[..., a] for a in range(3)]
    nz, ny, nx = vol.shape
    z = np.clip(z, 0, nz - 1.001); y = np.clip(y, 0, ny - 1.001); x = np.clip(x, 0, nx - 1.001)
    z0, y0, x0 = np.floor(z).astype(int), np.floor(y).astype(int), np.floor(x).astype(int)
    dz, dy, dx = z - z0, y - y0, x - x0
    v = 0.0
    for wz, iz in ((1 - dz, z0), (dz, z0 + 1)):
        for wy, iy in ((1 - dy, y0), (dy, y0 + 1)):
            for wx, ix in ((1 - dx, x0), (dx, x0 + 1)):
                v = v + wz * wy * wx * vol[np.clip(iz, 0, nz - 1), np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
    return v


def main(pid, topk=1):
    ct = load_mha(f"{ROOT}/{pid}/image/ct.mha")
    plan = json.load(open(f"{ROOT}/{pid}/{pid}.json"))
    hu_anchors = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json").hu_anchors
    dens = hu_to_density(ct.array, hu_anchors).astype(np.float32)
    body = binary_fill_holes(dens >= 0.1)
    pm = ProtonMachineData(device=DEV)
    rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]): (r["ray_source"], r["ray_target"], bl["energy"])
            for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}

    files = sorted(glob.glob(f"{CACHE}/{pid}/B*_R*_L*.npz"))
    # pick the beamlets with the largest MC dose (clearest Bragg peak)
    scored = []
    for f in files:
        d = np.load(f); scored.append((float(d["dose_max"]), f))
    scored.sort(reverse=True)
    ox, oy, oz = ct.origin; sx, sy, sz = ct.spacing

    for rank, (dm, f) in enumerate(scored[:topk]):
        name = os.path.basename(f)[:-4]
        b, r, l = [int(x[1:]) for x in name.split("_")]
        npz = np.load(f); bbox = npz["bbox"]; mc = npz["dose"].astype(np.float32); e = float(npz["energy"])
        src, tgt, energy = rays[(b, r, l)]
        z0, z1, y0, y1, x0, x1 = bbox
        kw = dict(out_bbox=tuple(bbox), machine=pm, density_override=dens, device=DEV)
        cur = proton_pb_dose_gpu(ct, src, tgt, energy, **kw)                                  # current
        msk = proton_pb_dose_gpu(ct, src, tgt, energy, mask_air=True, body_mask=body, **kw)   # mask_air
        skn = proton_pb_dose_gpu_skinentry(ct, src, tgt, energy, out_bbox=tuple(bbox),        # skin-entry
                                           machine=pm, density_override=dens, device=DEV)

        # ---- depth-dose profile along the beamlet axis through the MC peak ----
        axis = np.asarray(tgt, float) - np.asarray(src, float); axis /= np.linalg.norm(axis)
        pk = np.unravel_index(int(np.argmax(mc)), mc.shape)                    # peak voxel (local)
        pw = np.array([ox + (x0 + pk[2]) * sx, oy + (y0 + pk[1]) * sy, oz + (z0 + pk[0]) * sz])  # world
        tt = np.arange(-80, 80, 0.5)                                          # mm along axis about peak
        wpts = pw[None, :] + tt[:, None] * axis[None, :]                      # world xyz
        # world -> local bbox voxel idx (i=z,j=y,k=x)
        ijk = np.stack([(wpts[:, 2] - oz) / sz - z0, (wpts[:, 1] - oy) / sy - y0, (wpts[:, 0] - ox) / sx - x0], -1)
        prof = {k: trilerp(v, ijk) for k, v in [("MC", mc), ("current", cur), ("mask", msk), ("skin-entry", skn)]}
        dens_c = dens[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        prof_rho = trilerp(dens_c, ijk)

        # ---- figure ----
        zc = pk[0]
        fig, ax = plt.subplots(2, 4, figsize=(17, 8))
        vmax = max(mc.max(), cur.max(), skn.max())
        for j, (t2, v) in enumerate([("MC (GT)", mc), ("current", cur), ("mask_air", msk), ("skin-entry", skn)]):
            im = ax[0, j].imshow(v[zc], vmin=0, vmax=vmax, cmap="magma"); ax[0, j].set_title(t2)
            ax[0, j].contour(dens_c[zc], levels=[0.05], colors="cyan", linewidths=0.7)  # skin outline
        ax[1, 0].imshow(dens_c[zc], cmap="gray"); ax[1, 0].set_title("density (skin=cyan)")
        ax[1, 0].contour(dens_c[zc], levels=[0.05], colors="cyan", linewidths=0.7)
        axp = ax[1, 1]
        for k, c in [("MC", "k"), ("current", "tab:orange"), ("mask", "tab:green"), ("skin-entry", "tab:red")]:
            axp.plot(tt, prof[k] / (prof[k].max() + 1e-9), c, label=k, lw=1.6)
        axp.axvline(0, color="gray", ls=":"); axp.set_xlabel("mm along beam axis (0=MC peak)")
        axp.set_ylabel("dose (norm)"); axp.legend(fontsize=8); axp.set_title("depth-dose (Bragg peak)")
        axp2 = ax[1, 2]
        axp2.plot(tt, prof_rho, "b"); axp2.set_title("density along axis"); axp2.set_xlabel("mm"); axp2.axvline(0, color="gray", ls=":")
        # peak-shift summary
        def peakpos(p): return tt[int(np.argmax(p))]
        txt = (f"{pid} {name}  E={energy:.1f}\nMC dose_max={dm:.3g}\n\nBragg-peak position (mm, 0=MC):\n"
               f"  MC:         {peakpos(prof['MC']):+.1f}\n  current:    {peakpos(prof['current']):+.1f}\n"
               f"  mask_air:   {peakpos(prof['mask']):+.1f}\n  skin-entry: {peakpos(prof['skin-entry']):+.1f}")
        ax[1, 3].axis("off"); ax[1, 3].text(0.0, 0.9, txt, va="top", fontsize=10, family="monospace")
        for a in ax[0]: a.axis("off")
        plt.tight_layout()
        out = f"/home/kaiwang/doserad2026_workdir/runs/cv_eval/skinentry_verify_{pid}_{name}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
        print(f"[{rank}] {name} E={energy:.1f}: MC peak {peakpos(prof['MC']):+.1f} | cur {peakpos(prof['current']):+.1f} "
              f"| mask {peakpos(prof['mask']):+.1f} | skin {peakpos(prof['skin-entry']):+.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1ABB006", int(sys.argv[2]) if len(sys.argv) > 2 else 3)
