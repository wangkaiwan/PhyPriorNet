"""Lever-4 stage 1->2 bridge: turn the trained tissue classifier into a COARSE CT prior.

For each case: MR -> classifier (sliding-window) -> per-voxel class -> representative HU per class
(rep_hu from the manifest classes) -> coarse CT volume saved as .nii.gz. This coarse CT conditions the
stage-2 whole-image refiner (input = [MR, coarse CT]); the refiner learns the fine HU residual on top
of a correctly-*located* tissue map instead of hallucinating bone/lung density from scratch.

    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/precompute_coarse_ct.py \
        --clf $WORKDIR/sct_runs/clf/best.pt \
        --out $WORKDIR/cache/coarse_ct
"""
from __future__ import annotations

import os
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from train_sct_paired import norm_mr, load_arr
from train_sct_classifier import model, N_CLS

DATA = (os.environ.get("WORKDIR", "./workdir") + "/sct_data_2mm.json")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA); ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--soft", action="store_true", help="probability-weighted expected HU (smooth prior) "
                    "instead of argmax->rep_hu (hard 4-level)")
    ap.add_argument("--whole-image", action="store_true", help="whole-volume forward (match whole-image "
                    "classifier) instead of sliding-window")
    ap.add_argument("--rep-hu", default=None, help="comma per-class representative HU (override manifest; "
                    "needed for finer-bin classifiers). len = #classes.")
    a = ap.parse_args(); dev = "cuda"
    D = json.load(open(a.data))
    rep = (np.asarray([float(x) for x in a.rep_hu.split(",")], np.float32)
           if a.rep_hu else np.asarray(D["classes"]["rep_hu"], np.float32))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    net = model(len(rep)).to(dev).eval()            # n_cls from rep_hu (supports finer bins)
    net.load_state_dict(torch.load(a.clf, map_location=dev)["net"])
    import torch.nn.functional as F
    pad16 = lambda n: (16 - n % 16) % 16
    refp = set(D["refiner_pids"])                    # only what the 0.35T refiner needs (+ val)
    items = [it for it in D["train"] if it["pid"] in refp] + D["val"]
    print(f"generating coarse CT for {len(items)} cases (rep_hu={rep.tolist()}) -> {out}")
    for k, it in enumerate(items):
        pid = it["pid"]; ref = sitk.ReadImage(it["mr"])           # geometry reference (MR grid)
        mr = torch.from_numpy(norm_mr(load_arr(it["mr"])).astype(np.float32))[None, None].to(dev)
        with torch.autocast("cuda"):
            if a.whole_image:
                Z, Y, X = mr.shape[-3:]
                xp = F.pad(mr, (0, pad16(X), 0, pad16(Y), 0, pad16(Z)))
                logits = net(xp)[..., :Z, :Y, :X]
            else:
                logits = sliding_window_inference(mr, tuple(a.patch), 4, net, overlap=0.25, mode="gaussian")
        if a.soft:                                                 # E[HU] = sum_c softmax_c * rep_hu_c
            p = torch.softmax(logits.float(), dim=1)[0]            # (C,Z,Y,X)
            rep_t = torch.from_numpy(rep).to(p.device).view(-1, 1, 1, 1)
            coarse = (p * rep_t).sum(0).cpu().numpy()
        else:
            cls = logits.argmax(1)[0].cpu().numpy().astype(np.int64)   # hard 4-level
            coarse = rep[cls]
        # load_arr transposes (2,1,0); undo to write in sitk (z,y,x) order matching ref
        arr = np.transpose(coarse, (2, 1, 0)).astype(np.float32)
        im = sitk.GetImageFromArray(arr); im.CopyInformation(ref)
        sitk.WriteImage(im, str(out / f"{pid}.nii.gz"))
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(items)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
