"""Shared qualitative GT-vs-prediction plan visualization.

One figure per case: rows = views (axial/coronal/sagittal centred on the high-dose
centroid), cols = CT+isodose, GT, prediction, signed diff (%Rx), DTA (mm), local
gamma 1%/1mm (fail = red). MPLBACKEND=Agg (no display). Used by both
`scripts/visualize_case.py` (standalone) and `scripts/final_eval.py` (auto, inline).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doserad.eval.gamma import gamma_array


def _slice(a, view, i):
    """2-D slice of a (z,y,x) volume for axial/coronal/sagittal."""
    return a[i] if view == "axial" else (a[:, i, :] if view == "coronal" else a[:, :, i])


def render_sct_figure(*, patient, mri, sct, ct, pred, gt, g1, mask, rx, out,
                      views="axial,coronal,sagittal"):
    """Route-A sCT diagnostic figure. Cols (L→R): MRI | predicted dose on sCT | GT dose on CT |
    dose diff (pred−GT, %Rx) | local γ 1%/1mm. Rows = views, centred on the high-dose centroid."""
    views = [v.strip() for v in views.split(",")] if isinstance(views, str) else list(views)
    hi = gt > 0.5 * rx
    zc, yc, xc = ([int(round(c)) for c in np.array(np.where(hi)).mean(axis=1)]
                  if hi.any() else [s // 2 for s in gt.shape])
    idx = {"axial": zc, "coronal": yc, "sagittal": xc}
    cols = ["MRI", "pred dose (on sCT)", "GT dose (on CT)", "sCT − CT (HU)", "γ 1%/1mm"]
    fig, axs = plt.subplots(len(views), len(cols), figsize=(3.2 * len(cols), 3.2 * len(views)),
                            squeeze=False)
    for ri, view in enumerate(views):
        i = idx[view]; org = "upper" if view == "axial" else "lower"
        mr_s, sc_s, ct_s = _slice(mri, view, i), _slice(sct, view, i), _slice(ct, view, i)
        pr, gt_s = _slice(pred, view, i), _slice(gt, view, i)
        mk = _slice(mask, view, i); g1v = _slice(g1, view, i)
        dose_m = gt_s > 0.05 * rx; eval_m = (gt_s > 0.10 * rx) & mk
        for ci, (ax, title) in enumerate(zip(axs[ri], cols)):
            if ci == 0:
                ax.imshow(mr_s, cmap="gray", origin=org)
            elif ci in (1, 2):
                base = sc_s if ci == 1 else ct_s
                dose = pr if ci == 1 else gt_s
                ax.imshow(base, cmap="gray", origin=org, vmin=-200, vmax=300)
                im = ax.imshow(np.where(dose_m, dose, np.nan), cmap="jet", vmin=0, vmax=rx,
                               alpha=0.6, origin=org); plt.colorbar(im, ax=ax, fraction=.046)
            elif ci == 3:                                      # sCT − CT image error (HU), in-body
                body = ct_s > -500
                im = ax.imshow(np.where(body, sc_s - ct_s, np.nan), cmap="bwr", vmin=-300, vmax=300,
                               origin=org); plt.colorbar(im, ax=ax, fraction=.046)
            else:
                im = ax.imshow(np.where(eval_m, np.clip(g1v, 0, 2), np.nan), cmap="RdYlGn_r",
                               vmin=0, vmax=2, alpha=0.85, origin=org); plt.colorbar(im, ax=ax, fraction=.046)
            if ri == 0:
                ax.set_title(title, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{view} {i}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    gp = float((g1[mask] <= 1).mean() * 100) if mask.any() else float("nan")
    fig.suptitle(f"{patient}  sCT-route  |  γ1%/1mm pass={gp:.1f}%  Rx={rx:.3g}Gy", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return gp


def render_plan_figure(*, patient, ctarr, sp, gt, pred, g1, mask, rx, out,
                       dta=None, dta_search=10.0, views="axial,coronal,sagittal", cp=None):
    """Render the 6-panel GT-vs-pred figure to ``out`` (PNG). Returns γ1%/1mm pass %.

    Reuses already-computed ``g1``/``mask`` (the 1%/1mm gamma array + eval mask). ``dta``
    (mm) is computed once via a single DTA-limited gamma pass if not supplied — this is the
    only extra compute over a normal eval.
    """
    if dta is None:                                                            # distance-to-agreement map (mm)
        # STRICT dose tolerance (1% of Rx) so the gamma minimum is driven by DISTANCE: at each voxel,
        # how far to the nearest GT voxel whose dose matches within 1%. (A loose dose tol collapses
        # DTA to 0 — the min sits at the same voxel.) gamma*dta_search ≈ that distance, capped.
        gd, _ = gamma_array(pred, gt, sp, rx, dose_pct=1.0, dta_mm=dta_search)
        dta = np.clip(gd, 0, 1) * dta_search                                   # mm, capped at dta_search
    diff = (pred - gt) / rx * 100.0                                            # signed % of Rx

    views = [v.strip() for v in views.split(",")] if isinstance(views, str) else list(views)
    hi = gt > 0.5 * rx                                         # center each view on the high-dose centroid
    zc, yc, xc = ([int(round(c)) for c in np.array(np.where(hi)).mean(axis=1)]
                  if hi.any() else [s // 2 for s in gt.shape])
    idx = {"axial": zc, "coronal": yc, "sagittal": xc}
    cols = ["CT + iso (GT—/pred··)", "GT (Gy)", "pred (Gy)", "diff (%Rx)", "DTA (mm)", "γ 1%/1mm"]
    iso_lv = [0.3 * rx, 0.6 * rx, 0.9 * rx]; iso_c = ["c", "y", "r"]
    fig, axs = plt.subplots(len(views), len(cols), figsize=(3.2 * len(cols), 3.2 * len(views)),
                            squeeze=False)
    AL = 0.6                                                   # dose-overlay opacity
    for ri, view in enumerate(views):
        i = idx[view]; org = "upper" if view == "axial" else "lower"   # superior-up for cor/sag
        cts, gts_, prs, dfs = (_slice(ctarr, view, i), _slice(gt, view, i),
                               _slice(pred, view, i), _slice(diff, view, i))
        mk = _slice(mask, view, i); g1v = _slice(g1, view, i); dtv = _slice(dta, view, i)
        dose_m = gts_ > 0.05 * rx; eval_m = (gts_ > 0.10 * rx) & mk
        gov = np.where(dose_m, gts_, np.nan); pov = np.where(dose_m, prs, np.nan)
        dfo = np.where(eval_m, dfs, np.nan); dtas = np.where(eval_m, dtv, np.nan)
        g1s = np.where(eval_m, np.clip(g1v, 0, 2), np.nan)
        for ci, (ax, title) in enumerate(zip(axs[ri], cols)):
            if ci <= 2:
                ax.imshow(cts, cmap="gray", origin=org)        # CT under CT + the two DOSE panels
            if ci == 0:                                         # CT + isodose: GT solid, pred dashed
                ax.contour(gts_, levels=iso_lv, colors=iso_c, linewidths=0.7)
                ax.contour(prs, levels=iso_lv, colors=iso_c, linewidths=0.7, linestyles="dashed")
            elif ci == 1:                                       # GT colorwash only
                im = ax.imshow(gov, cmap="jet", vmin=0, vmax=rx, alpha=AL, origin=org); plt.colorbar(im, ax=ax, fraction=.046)
            elif ci == 2:                                       # pred colorwash only
                im = ax.imshow(pov, cmap="jet", vmin=0, vmax=rx, alpha=AL, origin=org); plt.colorbar(im, ax=ax, fraction=.046)
            elif ci == 3: im = ax.imshow(dfo, cmap="bwr", vmin=-5, vmax=5, alpha=0.7, origin=org); plt.colorbar(im, ax=ax, fraction=.046)
            elif ci == 4: im = ax.imshow(dtas, cmap="viridis", vmin=0, vmax=5, alpha=0.7, origin=org); plt.colorbar(im, ax=ax, fraction=.046)
            else:
                im = ax.imshow(g1s, cmap="RdYlGn_r", vmin=0, vmax=2, alpha=0.7, origin=org); plt.colorbar(im, ax=ax, fraction=.046)  # >1 = fail (red)
            if ri == 0: ax.set_title(title, fontsize=9)
            if ci == 0: ax.set_ylabel(f"{view} {i}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    gp = float((g1[mask] <= 1).mean() * 100) if mask.any() else float("nan")
    fig.suptitle(f"{patient}  {'plan' if not cp else cp}  |  γ1%/1mm pass={gp:.1f}%  Rx={rx:.3g}Gy",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)                                            # free fig memory (matters in a 16-patient loop)
    return gp
