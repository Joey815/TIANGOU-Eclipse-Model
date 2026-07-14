from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LimbProfile:
    alpha_rad: np.ndarray
    theta_rad: np.ndarray
    observer_distance_km: float
    sub_observer_lon_deg: float
    sub_observer_lat_deg: float

    @property
    def max_alpha_rad(self) -> float:
        return float(np.nanmax(self.alpha_rad))


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _profile_rotation(sub_observer_lon_deg: float, sub_observer_lat_deg: float) -> np.ndarray:
    lon = np.deg2rad(sub_observer_lon_deg)
    lat = np.deg2rad(sub_observer_lat_deg)
    return _rotation_y(lat) @ _rotation_z(-lon)


def _circular_fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(values)
    if valid.all():
        return values
    if not np.any(valid):
        raise ValueError("No valid limb samples were produced.")

    n = values.size
    idx = np.arange(n)
    valid_idx = idx[valid]
    valid_values = values[valid]

    ext_idx = np.concatenate([valid_idx - n, valid_idx, valid_idx + n])
    ext_values = np.concatenate([valid_values, valid_values, valid_values])
    filled = np.interp(idx, ext_idx, ext_values)
    return filled


def _update_max_alpha_from_points(
    max_alpha: np.ndarray,
    moon_points_xyz_km: np.ndarray,
    observer_distance_km: float,
    rot: np.ndarray,
    n_bins: int,
) -> int:
    moon_points_xyz_km = np.asarray(moon_points_xyz_km, dtype=float)
    if moon_points_xyz_km.size == 0:
        return 0
    if moon_points_xyz_km.ndim != 2 or moon_points_xyz_km.shape[1] != 3:
        raise ValueError("moon_points_xyz_km must have shape (N, 3)")

    pts = (rot @ moon_points_xyz_km.T).T
    rr = np.hypot(pts[:, 1], pts[:, 2])
    alpha = np.arctan2(rr, observer_distance_km - pts[:, 0])
    theta = np.mod(np.arctan2(pts[:, 1], pts[:, 2]), 2.0 * np.pi)

    indices = np.floor(theta / (2.0 * np.pi) * n_bins).astype(np.int64)
    indices = np.clip(indices, 0, n_bins - 1)
    np.maximum.at(max_alpha, indices, alpha)
    return int(moon_points_xyz_km.shape[0])


def build_limb_profile_from_point_blocks(
    moon_point_blocks_xyz_km,
    observer_distance_km: float,
    sub_observer_lon_deg: float,
    sub_observer_lat_deg: float,
    n_bins: int = 18000,
) -> LimbProfile:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    rot = _profile_rotation(sub_observer_lon_deg, sub_observer_lat_deg)
    max_alpha = np.full(n_bins, -np.inf, dtype=float)

    for block in moon_point_blocks_xyz_km:
        _update_max_alpha_from_points(
            max_alpha=max_alpha,
            moon_points_xyz_km=block,
            observer_distance_km=observer_distance_km,
            rot=rot,
            n_bins=n_bins,
        )

    max_alpha = _circular_fill(max_alpha)
    theta_centers = (np.arange(n_bins, dtype=float) + 0.5) * (2.0 * np.pi / n_bins)
    return LimbProfile(
        alpha_rad=max_alpha,
        theta_rad=theta_centers,
        observer_distance_km=float(observer_distance_km),
        sub_observer_lon_deg=float(sub_observer_lon_deg),
        sub_observer_lat_deg=float(sub_observer_lat_deg),
    )
