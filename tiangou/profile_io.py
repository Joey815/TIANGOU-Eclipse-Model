from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .lunar_limb import LimbProfile
from .spice_geometry import EclipseGeometry


@dataclass(frozen=True)
class CachedLimbProfile:
    profile: LimbProfile
    geometry: EclipseGeometry | None
    source_path: Path
    elapsed_s: float | None
    metadata: dict[str, Any]


def _maybe_scalar(value: np.ndarray) -> Any:
    if getattr(value, "shape", ()) == ():
        return value.item()
    return value


def load_limb_profile_npz(path: str | Path) -> CachedLimbProfile:
    source_path = Path(path).expanduser().resolve()
    with np.load(source_path, allow_pickle=False) as data:
        alpha_rad = np.asarray(data["alpha_rad"], dtype=float)
        theta_rad = np.asarray(data["theta_rad"], dtype=float)
        observer_distance_km = float(_maybe_scalar(data["observer_distance_km"]))
        sub_observer_lon_deg = float(_maybe_scalar(data["sub_observer_lon_deg"]))
        sub_observer_lat_deg = float(_maybe_scalar(data["sub_observer_lat_deg"]))

        profile = LimbProfile(
            alpha_rad=alpha_rad,
            theta_rad=theta_rad,
            observer_distance_km=observer_distance_km,
            sub_observer_lon_deg=sub_observer_lon_deg,
            sub_observer_lat_deg=sub_observer_lat_deg,
        )

        geometry = None
        geometry_raw = data["geometry"].item() if "geometry" in data.files else None
        if geometry_raw:
            geometry = EclipseGeometry(**json.loads(geometry_raw))

        elapsed_s = float(_maybe_scalar(data["elapsed_s"])) if "elapsed_s" in data.files else None
        metadata = {
            key: _maybe_scalar(data[key])
            for key in data.files
            if key
            not in {
                "alpha_rad",
                "theta_rad",
                "geometry",
                "observer_distance_km",
                "sub_observer_lon_deg",
                "sub_observer_lat_deg",
                "elapsed_s",
            }
        }

    return CachedLimbProfile(
        profile=profile,
        geometry=geometry,
        source_path=source_path,
        elapsed_s=elapsed_s,
        metadata=metadata,
    )
