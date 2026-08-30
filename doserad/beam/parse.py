"""Load a photon beam-config JSON into validated pydantic models."""
from __future__ import annotations

import json
from pathlib import Path

from doserad.beam.photon_config import PhotonPlan


def load_photon_plan(json_path: str | Path) -> PhotonPlan:
    data = json.loads(Path(json_path).read_text())
    return PhotonPlan.model_validate(data)
