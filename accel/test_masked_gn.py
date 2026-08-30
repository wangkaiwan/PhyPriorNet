"""Equivalence test: masked-GN padded-batch forward == per-sample unpadded forward.

Loads the deployed photon net + 4 real different-size CP crops; compares (a) plain per-sample
forward vs (b) one padded batch under valid_extents. PASS criteria: fp32 max|rel diff| < 1e-4
of each crop's dose max; autocast < 2e-3 (half precision noise). Also sanity: without the
context, MaskedGroupNorm must reproduce the original net exactly (0 diff).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doserad.data.dataset import normalize_channels  # noqa: E402
from doserad.model.unet3d import DoseUNet3D  # noqa: E402
from accel.masked_gn import install_masked_batching, valid_extents  # noqa: E402

DEV = "cuda"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_ssd/1ABB006")


def load_crops(n=4, max_vox=1_500_000):
    files = sorted(CACHE.glob("*.npz"), key=lambda f: -float(np.load(f)["dose_max"]))
    # exactness test only needs DIFFERENT sizes, not big ones — filter by crop volume
    small = [f for f in files if np.prod(np.load(f)["channels"].shape[1:]) < max_vox]
    picks = [small[0], small[len(small) // 3], small[2 * len(small) // 3], small[-1]][:n]
    xs = []
    for f in picks:
        z = np.load(f)
        inp = normalize_channels(z["channels"].astype(np.float32), add_naive=True,
                                 naive_skin_gate=True)
        xs.append(torch.from_numpy(inp))
    return xs


def main():
    net = DoseUNet3D(in_ch=6, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    st = torch.load("/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt",
                    map_location=DEV)
    net.load_state_dict(st["ema"])
    xs = load_crops()
    print("crop sizes:", [tuple(x.shape[-3:]) for x in xs])
    mod1 = torch.zeros(1, dtype=torch.long, device=DEV)

    def single(x, amp):
        d, h, w = x.shape[-3:]
        pad = (0, (-w) % 16, 0, (-h) % 16, 0, (-d) % 16)
        xin = torch.nn.functional.pad(x[None].to(DEV), pad)
        with torch.no_grad(), torch.autocast("cuda", enabled=amp):
            y = net(xin, mod1)
        return y[0, 0, :d, :h, :w].float()

    # reference outputs BEFORE swap
    for amp in (False, True):
        refs = [single(x, amp) for x in xs]
        if not amp:
            refs_fp32 = refs
        else:
            refs_amp = refs

    nswap = install_masked_batching(net)
    print(f"swapped {nswap} GroupNorms")

    # sanity: no context -> identical to original
    post = [single(x, False) for x in xs]
    d0 = max(float((a - b).abs().max()) for a, b in zip(refs_fp32, post))
    print(f"no-context sanity max|diff| = {d0:.2e}  (must be 0)")

    # padded batch with context
    for amp, refs, tol in ((False, refs_fp32, 1e-4), (True, refs_amp, 2e-3)):
        dims = [x.shape[-3:] for x in xs]
        D = -(-max(d[0] for d in dims) // 16) * 16
        H = -(-max(d[1] for d in dims) // 16) * 16
        W = -(-max(d[2] for d in dims) // 16) * 16
        xb = torch.zeros((len(xs), xs[0].shape[0], D, H, W), device=DEV)
        for i, x in enumerate(xs):
            d, h, w = x.shape[-3:]; xb[i, :, :d, :h, :w] = x.to(DEV)
        r16 = lambda v: -(-v // 16) * 16
        ext16 = [(r16(x.shape[-3]), r16(x.shape[-2]), r16(x.shape[-1])) for x in xs]
        with valid_extents(ext16, full=(D, H, W)):
            with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                yb = net(xb, torch.zeros(len(xs), dtype=torch.long, device=DEV))
        worst = 0.0
        for i, (x, ref) in enumerate(zip(xs, refs)):
            d, h, w = x.shape[-3:]
            got = yb[i, 0, :d, :h, :w].float()
            rel = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-9))
            worst = max(worst, rel)
        status = "PASS" if worst < tol else "FAIL"
        print(f"batched-vs-single max rel diff ({'amp' if amp else 'fp32'}): {worst:.2e}  "
              f"tol {tol:.0e}  -> {status}")


if __name__ == "__main__":
    main()
