"""Validate the GPU proton PB engine (doserad/physics/proton_pb_gpu.py) against the verified
pyRadPlan Hong-PB reference (scripts/proton_pb_beamlet.py wrapper) for several beamlets spanning
the energy range, on patient 1ABB006.

TWO-PHASE (different conda envs):

  Phase 1 (pyradplan env) -- compute reference PB dose .mha per beamlet on a fixed bbox:
      CUDA_VISIBLE_DEVICES=1 conda run -n pyradplan --no-capture-output python -u \
          scripts/validate_proton_pb_gpu.py --phase ref --out /tmp/proton_pb_ref

  Phase 2 (doserad env) -- run the GPU engine on the SAME bbox & compare (corr, rel-err, timing):
      CUDA_VISIBLE_DEVICES=1 conda run -n doserad --no-capture-output python -u \
          scripts/validate_proton_pb_gpu.py --phase gpu --ref /tmp/proton_pb_ref
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np

ROOT = "/data/kwang/DoseRad2026_raw"
PID, MODALITY, SPLIT = "1ABB006", "proton", "training"
MARGIN = 4
THRESH = 0.01


def _bbox_from_dose(dose, margin, shape):
    nz = np.argwhere(dose > THRESH * dose.max())
    if len(nz) == 0:
        return None
    z0, y0, x0 = nz.min(0); z1, y1, x1 = nz.max(0)
    return (max(int(z0) - margin, 0), min(int(z1) + margin, shape[0] - 1),
            max(int(y0) - margin, 0), min(int(y1) + margin, shape[1] - 1),
            max(int(x0) - margin, 0), min(int(x1) + margin, shape[2] - 1))


def _select_beamlets(plan, n_want=12):
    """Pick beamlets spanning the energy range (beam0/ray0..., sorted by energy)."""
    items = []
    for beam in plan["beams"]:
        for ray in beam["rays"]:
            for bl in ray["beamlets"]:
                items.append((beam, ray, bl))
    items.sort(key=lambda t: t[2]["energy"])
    # evenly sample across sorted energy
    idxs = np.linspace(0, len(items) - 1, n_want).round().astype(int)
    return [items[i] for i in sorted(set(idxs.tolist()))]


# ---------------------------------------------------------------- PHASE 1: pyRadPlan reference
def phase_ref(out_dir):
    import SimpleITK as sitk
    sys.path.insert(0, "/home/kaiwang/project/pyradplan-pb-baseline")
    import baseline_protons as B
    from pyRadPlan.machines import load_from_name

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    base = Path(ROOT) / MODALITY / SPLIT / PID
    ct = B.ct_from_file(str(base / B.DIR_IMAGE / B.CT_NAME))
    plan = json.load(open(base / f"{PID}.json"))
    beam_params = json.load(open(Path(ROOT) / B.BEAM_PARAMS_FILENAME))
    hlut = np.array([tuple(e.values()) for e in beam_params["hu_to_density"]["entries"]], dtype=float)
    machine = load_from_name("protons", "Generic")
    energies = np.array(sorted(machine.energies), dtype=np.float64)
    sad = machine.sad
    cst = B.StructureSet(vois=[], ct_image=ct); cst.create_body_seg()
    pln = B.IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_dose_calc = {"dose_grid": {"resolution": ct.resolution}, "air_offset_correction": True,
                          "geometric_lateral_cutoff": 25.0, "trace_on_dose_grid": True, "hlut": hlut}

    sel = _select_beamlets(plan)
    full_shape = sitk.GetArrayFromImage(ct.cube_hu if hasattr(ct, "cube_hu") else
                                        sitk.ReadImage(str(base / B.DIR_IMAGE / B.CT_NAME))).shape
    meta = []
    for beam, ray, bl in sel:
        b, r, l = beam["beam_idx"], ray["ray_idx"], bl["beamlet_idx"]
        t = time.time()
        pb_dose, energy = B._compute_beamlet_dose(ct, cst, pln, beam, ray, bl, sad, energies)
        arr = sitk.GetArrayFromImage(pb_dose).astype(np.float32)
        bbox = _bbox_from_dose(arr, MARGIN, arr.shape)
        if bbox is None:
            print(f"  B{b}R{r}L{l} E={energy:.1f} -> empty, skip", flush=True); continue
        z0, z1, y0, y1, x0, x1 = bbox
        crop = arr[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        fn = out / f"ref_B{b}_R{r}_L{l}.npz"
        np.savez_compressed(fn, dose=crop, bbox=np.asarray(bbox, np.int32),
                            energy=np.float32(energy), json_energy=np.float32(bl["energy"]),
                            ray_source=np.asarray(ray["ray_source"], np.float64),
                            ray_target=np.asarray(ray["ray_target"], np.float64),
                            b=b, r=r, l=l)
        meta.append((b, r, l, float(energy)))
        print(f"  B{b}R{r}L{l} E={energy:.2f}MeV bbox={bbox} max={crop.max():.3g} "
              f"[{time.time()-t:.1f}s] -> {fn.name}", flush=True)
    print(f"REF DONE: {len(meta)} beamlets", flush=True)


# ---------------------------------------------------------------- PHASE 2: GPU engine + compare
def phase_gpu(ref_dir):
    import torch
    from doserad.io.mha import load_mha
    from doserad.physics.machine import load_photon_machine
    from doserad.physics.density import hu_to_density
    from doserad.physics.proton_pb_gpu import ProtonMachineData, proton_pb_dose_gpu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path(ROOT) / MODALITY / SPLIT / PID
    ct = load_mha(base / "image" / "ct.mha")
    machine = load_photon_machine(str(Path(ROOT) / "beam_parameters.json"))
    pm = ProtonMachineData(device=dev)
    # Density is constant across all beamlets of a patient -> compute ONCE (40M-voxel
    # hu_to_density is ~475ms; doing it per beamlet would dominate runtime).
    density = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)

    refs = sorted(Path(ref_dir).glob("ref_*.npz"), key=lambda p: float(np.load(p)["energy"]))
    print(f"device={dev}  comparing {len(refs)} beamlets\n", flush=True)
    print(f"{'beamlet':14} {'E(MeV)':>7} {'corr':>7} {'relL2':>7} {'gpuMax':>9} {'refMax':>9} "
          f"{'ratio':>6} {'ms':>7}")
    corrs = []
    for rf in refs:
        d = np.load(rf)
        ref = d["dose"].astype(np.float32)
        bbox = tuple(int(v) for v in d["bbox"])
        e = float(d["energy"]); src = d["ray_source"]; tgt = d["ray_target"]
        # warmup (first call triggers cudnn autotune / kernel compile)
        _ = proton_pb_dose_gpu(ct, src, tgt, e, out_bbox=bbox, machine=pm,
                               density_override=density, device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        gpu = proton_pb_dose_gpu(ct, src, tgt, e, out_bbox=bbox, machine=pm,
                                 density_override=density, device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000

        m = ref > 0.01 * ref.max()
        a, b = gpu[m].ravel(), ref[m].ravel()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
        tag = f"B{int(d['b'])}R{int(d['r'])}L{int(d['l'])}"
        ratio = gpu.max() / (ref.max() + 1e-12)
        corrs.append(corr)
        print(f"{tag:14} {e:7.2f} {corr:7.3f} {rel:7.3f} {gpu.max():9.3g} {ref.max():9.3g} "
              f"{ratio:6.3f} {ms:7.1f}", flush=True)

    corrs = np.array(corrs)
    print(f"\nMEAN corr={np.nanmean(corrs):.3f}  min={np.nanmin(corrs):.3f}  "
          f"median={np.nanmedian(corrs):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ref", "gpu"], required=True)
    ap.add_argument("--out", default="/tmp/proton_pb_ref")
    ap.add_argument("--ref", default="/tmp/proton_pb_ref")
    a = ap.parse_args()
    if a.phase == "ref":
        phase_ref(a.out)
    else:
        phase_gpu(a.ref)


if __name__ == "__main__":
    main()
