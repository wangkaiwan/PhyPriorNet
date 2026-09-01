"""
doserad.metrics_beam
--------------------
Level 1 single-beam evaluation metrics.

Metrics
-------
masked_beam_mae     MAE restricted to the high-dose region, normalised by beam max.
idd_curve_distance  Normalised RMS difference of integrated depth-dose curves.
evaluate_beam_level Aggregate both metrics over a list of beams.
"""

import math

import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple


def masked_beam_mae(pred_beam: np.ndarray, gt_beam: np.ndarray) -> float:
    """Masked MAE for a single beam.

    Computed only in voxels where the ground-truth dose is >= 10 % of the
    beam's max dose, then normalised by that max dose.

    Returns
    -------
    float
        MAE as a fraction (multiply by 100 for a percentage).
        ``nan`` if the beam is empty or the mask contains no voxels.
    """
    beam_max = float(np.max(gt_beam))
    if beam_max <= 0:
        return float('nan')

    mask = gt_beam >= 0.1 * beam_max
    if not mask.any():
        return float('nan')

    return float(np.mean(np.abs(pred_beam[mask] - gt_beam[mask])) / beam_max)


def compute_idd_curve(dose_3d: np.ndarray, direction: np.ndarray,
                      spacing: np.ndarray) -> np.ndarray:
    """Integrated Depth-Dose (IDD) curve along beam direction.

    Value at each depth is the dose summed over the perpendicular plane.
    Samples span the in-plane diagonal, so the length is angle-independent.
    Summing over z first to reduce the resampling to 2D.
    """
    if abs(direction[2]) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")

    plane = dose_3d.sum(axis=0, dtype=np.float64)
    ny, nx = plane.shape
    sx, sy = float(spacing[0]), float(spacing[1])
    plane_centre = ((nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0)

    # square output, one sample per voxel, wide enough to hold the diagonal
    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    out_origin = (-(n - 1) * step / 2.0,) * 2  # puts the middle sample at 0

    source = sitk.GetImageFromArray(plane)
    source.SetSpacing((sx, sy))

    # Resample maps each output point back to the input, so turning by the beam
    # angle about the plane's centre lays the beam along the output's first axis
    to_beam = sitk.Euler2DTransform()
    to_beam.SetCenter((0.0, 0.0))
    to_beam.SetAngle(math.atan2(direction[1], direction[0]))
    to_beam.SetTranslation(plane_centre)
    aligned = sitk.Resample(source, (n, n), to_beam, sitk.sitkLinear,
                            out_origin, (step, step), (1.0, 0.0, 0.0, 1.0),
                            0.0, sitk.sitkFloat64)

    #collapsing lateral dose against depth
    return sitk.GetArrayFromImage(aligned).sum(axis=0)


def idd_curve_distance(pred_beam: np.ndarray, gt_beam: np.ndarray,
                       direction: np.ndarray, spacing: np.ndarray) -> float:
    """Normalised RMS difference between predicted and ground-truth IDD curves.

    Both curves are normalised by the GT IDD maximum before computing the RMS,
    so the result is dimensionless.

    Returns
    -------
    float
        Normalised RMS IDD distance, or ``nan`` if the GT curve is flat.
    """
    idd_pred = compute_idd_curve(pred_beam, direction, spacing)
    idd_gt   = compute_idd_curve(gt_beam,   direction, spacing)

    idd_max = float(np.max(idd_gt))
    if idd_max <= 0:
        return float('nan')

    return float(np.sqrt(np.mean((idd_pred / idd_max - idd_gt / idd_max) ** 2)))


def directions_of(plan: dict) -> dict:
    """(beam_idx, cp_idx | ray_idx) -> which way the beam travels."""
    out = {}
    for beam in plan["beams"]:
        bi = beam["beam_idx"]
        for cp in beam.get("control_points", []):  # photon
            g = math.radians(float(cp["gantry_angle"]))
            out[bi, cp["cp_idx"]] = np.array([-math.sin(g), math.cos(g), 0.0])
        for ray in beam.get("rays", []):  # proton
            v = (np.asarray(ray["ray_target"], float)
                 - np.asarray(ray["ray_source"], float))
            out[bi, ray["ray_idx"]] = v / np.linalg.norm(v)
    return out


def evaluate_beam_level(pred_beams: List[np.ndarray],
                        gt_beams: List[np.ndarray],
                        directions: List[np.ndarray],
                        spacing: np.ndarray) -> Dict:
    """Evaluate Level 1 metrics over all beams in a plan.

    Parameters
    ----------
    pred_beams:
        Predicted dose array for each beam.
    gt_beams:
        Ground-truth dose array for each beam (same ordering).
    directions:
        Beam propagation direction for each beam (same ordering), as used
        by compute_idd_curve/idd_curve_distance.
    spacing:
        Voxel spacing (x, y, z) shared by all beams in the plan.

    Returns
    -------
    dict with keys:
        beam_mae_per_beam, beam_mae_mean, beam_mae_std,
        idd_distance_per_beam, idd_distance_mean, idd_distance_std
    """
    maes:      List[float] = []
    idd_dists: List[float] = []

    for pred_b, gt_b, direction in zip(pred_beams, gt_beams, directions):
        maes.append(masked_beam_mae(pred_b, gt_b))
        idd_dists.append(idd_curve_distance(pred_b, gt_b, direction, spacing))

    return {
        'beam_mae_per_beam':      maes,
        'beam_mae_mean':          float(np.nanmean(maes)),
        'beam_mae_std':           float(np.nanstd(maes)),
        'idd_distance_per_beam':  idd_dists,
        'idd_distance_mean':      float(np.nanmean(idd_dists)),
        'idd_distance_std':       float(np.nanstd(idd_dists)),
    }
