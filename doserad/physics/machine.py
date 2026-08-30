"""Load the DoseRAD photon machine model (beam_parameters.json) — the same
linac/MC parameters used to generate the ground-truth dose."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhotonMachine:
    sad_mm: float
    num_leaf_pairs: int
    leaf_thickness_mm: float
    jaw_x_mm: tuple[float, float]
    jaw_y_mm: tuple[float, float]
    source_plane_distance_mm: float
    virtual_source_distance_mm: float
    spectrum_mev: tuple[float, ...]
    spectrum_weight: tuple[float, ...]
    hu_anchors: tuple[tuple[float, float], ...]

    def leaf_pair_y_bounds_mm(self, pair_idx: int) -> tuple[float, float]:
        half = self.num_leaf_pairs / 2.0
        lo = (pair_idx - half) * self.leaf_thickness_mm
        return (lo, lo + self.leaf_thickness_mm)


def load_photon_machine(path: str | Path) -> PhotonMachine:
    data = json.loads(Path(path).read_text())
    ph = data["photon"]
    spec = ph["energy_spectrum"]
    w = [float(x) for x in spec["weight"]]
    s = sum(w) or 1.0
    w = [x / s for x in w]
    anchors = sorted((float(e["hu"]), float(e["density_g_cm3"]))
                     for e in data["hu_to_density"]["entries"])
    return PhotonMachine(
        sad_mm=float(ph["SAD_mm"]),
        num_leaf_pairs=int(ph["mlc_num_leaf_pairs"]),
        leaf_thickness_mm=float(ph["mlc_leaf_thickness_mm"]),
        jaw_x_mm=(float(ph["jaw_x_mm"][0]), float(ph["jaw_x_mm"][1])),
        jaw_y_mm=(float(ph["jaw_y_mm"][0]), float(ph["jaw_y_mm"][1])),
        source_plane_distance_mm=float(ph["source_plane_distance_mm"]),
        virtual_source_distance_mm=float(ph["virtual_source_distance_mm"]),
        spectrum_mev=tuple(float(x) for x in spec["energy_mev"]),
        spectrum_weight=tuple(w),
        hu_anchors=tuple(anchors),
    )
