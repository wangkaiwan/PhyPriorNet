"""In-container MRI -> sCT density (shared by photon_mri and proton_mri).

Reproduces eval_dose_e2e / eval_protonmri_plan's synth stage EXACTLY (verified: coarse repro 0.000 HU):
  MR -> mr01 (percentile 1/99 norm)
  MR -> clf_whole (whole-image, softmax -> E[HU]) -> coarse CT     [rep_hu = -1000,-600,30,700]
  synth([mr01, coarse01]) -> sCT01  (E2E.synth, 2-ch MONAI UNet, INSTANCE norm)
  density = sct01*DENS_MAX          (density_direct, proton)
          | hu_to_density_torch(sct01*(CT_HI-CT_LO)+CT_LO)   (photon)

The synth net is trained at 2mm iso (the photon MR grid). `synth_density` resamples an arbitrary
input grid -> 2mm for synth, then resamples the density back to the input grid, so the container is
grid-robust (photon-MRI source == 2mm -> ~identity; proton-MRI source == native -> real resample).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from doserad.physics.diff_channels import hu_to_density_torch

CT_LO, CT_HI = -1000.0, 2000.0
DENS_MAX = 2.5
REP_HU = np.asarray([-1000.0, -600.0, 30.0, 700.0], np.float32)   # clf_whole 4-class rep HU
_SYNTH_SPACING = (2.0, 2.0, 2.0)                                   # synth training grid (x,y,z mm)
_pad16 = lambda n: (16 - n % 16) % 16


def load_classifier(clf_path, dev):
    from train_sct_classifier import model as clf_model
    net = clf_model(len(REP_HU)).to(dev).eval()
    net.load_state_dict(torch.load(clf_path, map_location=dev)["net"])
    return net


def _norm_mr01(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / max(hi - lo, 1.0), 0.0, 1.0).astype(np.float32)


def _resample(img, ref_grid, default, interp=sitk.sitkLinear):
    return sitk.Resample(img, ref_grid, sitk.Transform(), interp, default, sitk.sitkFloat32)


def _grid_at_spacing(ref_img, spacing):
    """A sitk grid covering ref_img's physical extent at the given isotropic spacing (same origin/dir)."""
    size = [int(round(ref_img.GetSize()[i] * ref_img.GetSpacing()[i] / spacing[i])) for i in range(3)]
    g = sitk.Image(size, sitk.sitkFloat32)
    g.SetOrigin(ref_img.GetOrigin()); g.SetDirection(ref_img.GetDirection()); g.SetSpacing(spacing)
    return g


@torch.no_grad()
def synth_density(mr_sitk, clf, synth, dev, density_direct: bool, hu_anchors=None, timings=None,
                  native_grid: bool = False):
    """mr_sitk: source MR (any grid). Returns (density_np (z,y,x) on the SOURCE grid, sct01 tensor).
    Runs the classifier+synth on a 2mm grid, then resamples density back to the source grid.

    native_grid=True: the synth was trained at the SOURCE resolution (proton native 1x1x3,
    native_synth E2E) -> run clf+synth directly on the source grid, NO 2mm resample either way.
    This matches precompute_coarse_ct.py + eval_proton_e2e_held16.py exactly (INSTANCE norm is
    extent-sensitive, so a 2mm-resampled input to a native-trained synth would be wrong).

    `timings` (optional dict) accumulates seconds per stage. The photon-MRI container is
    compute-bound (103 s on a job whose byte-identical photon-CT twin took 46 s, all of the extra
    being this function, ~11 s per source image), and the two SimpleITK resamples here run on 4
    vCPUs over ~10M voxels, so they are not obviously cheaper than the two GPU forwards. Splitting
    them is the only way to know which one to attack.
    """
    def _t(key, t0):
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + (time.time() - t0)

    src_spacing = mr_sitk.GetSpacing()
    at_2mm = all(abs(src_spacing[i] - 2.0) < 1e-3 for i in range(3))
    use_src = at_2mm or native_grid                      # run on the source grid, no resample
    t0 = time.time()
    mr2 = mr_sitk if use_src else _resample(mr_sitk, _grid_at_spacing(mr_sitk, _SYNTH_SPACING), 0.0)
    _t("resample_in", t0)

    a_mr = sitk.GetArrayFromImage(mr2).astype(np.float32)          # (z,y,x) 2mm
    mr01 = torch.from_numpy(_norm_mr01(a_mr)).to(dev)
    # classifier wants (x,y,z) axis order (train_sct_paired.load_arr transposes 2,1,0)
    mrx = torch.from_numpy(np.transpose(_norm_mr01(a_mr), (2, 1, 0)))[None, None].to(dev)
    t0 = time.time()
    # coarse recipe MUST match how it was generated (precompute_coarse_ct.py): the 2mm photon clf_whole
    # was WHOLE-IMAGE, but the native 1x1x3 coarse (clf_1x1x3_samefield) was SLIDING-WINDOW (128^3, 0.25).
    # Whole-image on native lung volumes mis-classifies (verified: lung dose gamma 90->48). Tie the recipe
    # to native_grid: native -> sliding, 2mm -> whole. (repro exact: best.pt+slide = 0.0 HU vs precomputed.)
    with torch.autocast("cuda", enabled=(dev != "cpu")):
        if native_grid:
            from monai.inferers import sliding_window_inference
            logits = sliding_window_inference(mrx, (128, 128, 128), 4, clf, overlap=0.25, mode="gaussian")
            p = torch.softmax(logits.float(), 1)[0]                        # (C,x,y,z)
        else:
            X, Y, Z = mrx.shape[-3:]
            xp = F.pad(mrx, (0, _pad16(Z), 0, _pad16(Y), 0, _pad16(X)))
            p = torch.softmax(clf(xp)[..., :X, :Y, :Z].float(), 1)[0]       # (C,x,y,z)
        coarse = (p * torch.from_numpy(REP_HU).to(dev).view(-1, 1, 1, 1)).sum(0)
        coarse = coarse.permute(2, 1, 0)                                    # -> (z,y,x)
        co = torch.clamp((coarse - CT_LO) / (CT_HI - CT_LO), 0, 1)
        if timings is not None and dev != "cpu":
            torch.cuda.synchronize()
        _t("clf", t0)
        t0 = time.time()
        sct01 = synth.sct01(torch.stack([mr01, co], 0)[None])[0, 0]         # (z,y,x) 2mm
        if density_direct:
            dens2 = (sct01.float().clamp(0, 1) * DENS_MAX)
        else:
            dens2 = hu_to_density_torch(sct01 * (CT_HI - CT_LO) + CT_LO, hu_anchors).float()
        if timings is not None and dev != "cpu":
            torch.cuda.synchronize()
        _t("synth", t0)

    if use_src:
        t0 = time.time()
        out = dens2.cpu().numpy().astype(np.float32)
        _t("resample_out", t0)
        return out, sct01
    t0 = time.time()
    dens_img = sitk.GetImageFromArray(dens2.cpu().numpy().astype(np.float32)); dens_img.CopyInformation(mr2)
    dens_src = _resample(dens_img, mr_sitk, 0.0)
    out = sitk.GetArrayFromImage(dens_src).astype(np.float32)
    _t("resample_out", t0)
    return out, sct01
