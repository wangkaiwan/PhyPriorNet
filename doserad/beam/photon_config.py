"""Pydantic models for the DoseRAD2026 photon beam JSON (verified schema)."""
from __future__ import annotations

from pydantic import BaseModel, model_validator


class ControlPoint(BaseModel):
    cp_idx: int
    gantry_angle: float
    mlc_left_int_mm: list[float]
    mlc_right_int_mm: list[float]

    @model_validator(mode="after")
    def _equal_leaf_counts(self) -> "ControlPoint":
        if len(self.mlc_left_int_mm) != len(self.mlc_right_int_mm):
            raise ValueError("left/right MLC leaf counts differ")
        return self


class PhotonBeam(BaseModel):
    beam_idx: int
    SAD: float
    num_mlc_leaf_pairs: int
    iso_center: tuple[float, float, float]
    control_points: list[ControlPoint]


class PhotonPlan(BaseModel):
    beams: list[PhotonBeam]

    @property
    def n_control_points(self) -> int:
        return sum(len(b.control_points) for b in self.beams)
