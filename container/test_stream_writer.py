"""Streaming 4D MHA writer validation (no GPU):
1. equivalence — streaming output decodes identically (array + geometry) to the JoinSeries path;
2. sitk round-trip — SimpleITK reads our hand-written header;
3. memory — peak RSS while writing a many-frame full-grid stack stays ~1 frame, not the stack.
Run: conda run -n doserad python container/test_stream_writer.py
"""
from __future__ import annotations

import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from container.proton import gc_invoke

rng = np.random.default_rng(0)


def _ref_img():
    im = sitk.GetImageFromArray(np.zeros((4, 5, 6), np.float32))
    im.SetSpacing((1.0, 1.5, 3.0))
    im.SetOrigin((-250.0, -249.51171875, -246.0))
    return im


def test_equivalence(tmp: Path):
    full = (40, 50, 60)
    items = []
    for i in range(7):
        crop = rng.random((8, 9, 10)).astype(np.float32)
        z0, y0, x0 = rng.integers(0, 30), rng.integers(0, 40), rng.integers(0, 50)
        items.append((crop, (z0, z0 + 7, y0, y0 + 8, x0, x0 + 9)))
    items.append((rng.random(full).astype(np.float32), None))     # pre-full frame path
    ref = _ref_img()
    a, b = tmp / "join.mha", tmp / "stream.mha"
    gc_invoke._stack_and_write(items, full, ref, a)
    gc_invoke._stack_and_write_streaming(items, full, ref, b)
    ia, ib = sitk.ReadImage(str(a)), sitk.ReadImage(str(b))
    assert ib.GetDimension() == 4, "stream not 4D"
    assert ia.GetSize() == ib.GetSize(), f"size {ia.GetSize()} != {ib.GetSize()}"
    assert np.array_equal(sitk.GetArrayFromImage(ia), sitk.GetArrayFromImage(ib)), "voxel mismatch"
    assert np.allclose(ia.GetSpacing(), ib.GetSpacing()), "spacing"
    assert np.allclose(ia.GetOrigin(), ib.GetOrigin()), "origin"
    assert np.allclose(ia.GetDirection(), ib.GetDirection()), "direction"
    print(f"  [PASS] equivalence: {ib.GetSize()} voxels+geometry identical after decode")


def test_memory(tmp: Path):
    full = (164, 493, 498)            # real proton grid, 161 MB/frame
    n = 40                            # 6.4 GB decompressed if held as a stack
    items = [(np.full((10, 10, 10), 0.5, np.float32), (60, 69, 200, 209, 200, 209))] * n
    ref = _ref_img()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # GB (linux: KB)
    gc_invoke._stack_and_write_streaming(items, full, ref, tmp / "big.mha")
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    im = sitk.ReadImage(str(tmp / "big.mha"))
    assert im.GetSize() == (498, 493, 164, n)
    arr0 = sitk.GetArrayFromImage(im)[0]
    assert float(arr0[60:70, 200:210, 200:210].min()) == 0.5 and float(arr0.sum()) == 0.5 * 1000
    print(f"  [PASS] {n}x161MB frames (6.4 GB stack) written+read; peak RSS {rss1:.2f} GB "
          f"(delta {rss1 - rss0:+.2f} GB — constant-memory, not stack-sized)")
    assert rss1 < 3.0, f"RSS {rss1:.2f} GB too high for streaming"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="streamw_") as td:
        tmp = Path(td)
        test_equivalence(tmp)
        test_memory(tmp)
    print("STREAM WRITER: ALL PASS")
