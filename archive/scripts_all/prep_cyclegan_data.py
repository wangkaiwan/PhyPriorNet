"""Convert MR/CT .mha pairs -> the 3D-CycleGAN .nii layout:
   <out>/images/<id>.nii.gz   (domain A = MRI)
   <out>/labels/<id>.nii.gz   (domain B = CT)

Leak-aware: --exclude-ids skips patients (used to drop DoseRAD-overlap cases from the
SynthRAD2025 TRAIN set, so validation on DoseRAD stays clean). CPU only.

DoseRAD layout : <root>/<pid>/image/{mr.mha,ct.mha}
SynthRAD layout: pass --src-glob to point at <pid> dirs and --mr-name/--ct-name.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing per-patient dirs")
    ap.add_argument("--out", required=True, help="output dir (gets images/ and labels/)")
    ap.add_argument("--ids", nargs="*", default=None, help="explicit patient ids; else all subdirs of root")
    ap.add_argument("--mr-rel", default="image/mr.mha", help="MR path relative to <root>/<id>/")
    ap.add_argument("--ct-rel", default="image/ct.mha", help="CT path relative to <root>/<id>/")
    ap.add_argument("--exclude-ids", nargs="*", default=[], help="ids to SKIP (leak exclusion)")
    args = ap.parse_args()

    root = Path(args.root)
    ids = args.ids or sorted(p.name for p in root.iterdir() if p.is_dir())
    excl = set(args.exclude_ids)
    imgd = Path(args.out) / "images"; labd = Path(args.out) / "labels"
    imgd.mkdir(parents=True, exist_ok=True); labd.mkdir(parents=True, exist_ok=True)

    n, skipped = 0, 0
    for pid in ids:
        if pid in excl:
            skipped += 1; continue
        mr = root / pid / args.mr_rel; ct = root / pid / args.ct_rel
        if not (mr.exists() and ct.exists()):
            continue
        sitk.WriteImage(sitk.ReadImage(str(mr)), str(imgd / f"{pid}.nii.gz"))
        sitk.WriteImage(sitk.ReadImage(str(ct)), str(labd / f"{pid}.nii.gz"))
        n += 1
    print(f"wrote {n} pairs to {args.out} (images=MRI, labels=CT); skipped {skipped} excluded")


if __name__ == "__main__":
    main()
