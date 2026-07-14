from __future__ import annotations

import numpy as np

from .lunar_limb import LimbProfile
from .spice_geometry import EclipseGeometry


_TWO_PI = 2.0 * np.pi


def _moon_center(geometry: EclipseGeometry) -> tuple[float, float]:
    angle = (geometry.sun_position_angle_rad + np.pi) % _TWO_PI
    east = geometry.sun_moon_separation_rad * np.sin(angle)
    north = geometry.sun_moon_separation_rad * np.cos(angle)
    return float(east), float(north)


def _limb_lookup(
    geometry: EclipseGeometry,
    profile: LimbProfile,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.mod(np.asarray(profile.theta_rad, dtype=float), _TWO_PI)
    radii = (
        profile.observer_distance_km / geometry.moon_distance_km
    ) * np.asarray(profile.alpha_rad, dtype=float)
    order = np.argsort(theta)
    theta = theta[order]
    radii = radii[order]
    return (
        np.concatenate([theta[-1:] - _TWO_PI, theta, theta[:1] + _TWO_PI]),
        np.concatenate([radii[-1:], radii, radii[:1]]),
    )


def terrain_limb_visible_map(
    shape: tuple[int, int],
    center_column: float,
    center_row: float,
    pixels_per_radian: float,
    geometry: EclipseGeometry,
    profile: LimbProfile,
) -> np.ndarray:
    theta_ext, radii_ext = _limb_lookup(geometry, profile)
    moon_east, moon_north = _moon_center(geometry)
    moon_column = center_column - moon_east * pixels_per_radian
    moon_row = center_row - moon_north * pixels_per_radian
    margin = int(np.ceil(float(np.nanmax(radii_ext)) * pixels_per_radian)) + 2

    row0 = max(0, int(np.floor(moon_row - margin)))
    row1 = min(shape[0], int(np.ceil(moon_row + margin)) + 1)
    col0 = max(0, int(np.floor(moon_column - margin)))
    col1 = min(shape[1], int(np.ceil(moon_column + margin)) + 1)

    visible = np.ones(shape, dtype=np.float32)
    if row0 >= row1 or col0 >= col1:
        return visible

    rows = np.arange(row0, row1, dtype=float)[:, None]
    cols = np.arange(col0, col1, dtype=float)[None, :]
    north = -(rows - center_row) / pixels_per_radian - moon_north
    east = -(cols - center_column) / pixels_per_radian - moon_east
    radius = np.hypot(east, north)
    theta = np.mod(
        np.arctan2(east, north) - geometry.moon_axis_angle_rad,
        _TWO_PI,
    )
    limb_radius = np.interp(theta.ravel(), theta_ext, radii_ext).reshape(theta.shape)
    visible[row0:row1, col0:col1] = (radius >= limb_radius).astype(np.float32)
    return visible


def terrain_limb_image_transmission(
    image: np.ndarray,
    center_column: float,
    center_row: float,
    pixels_per_radian: float,
    geometry: EclipseGeometry,
    profile: LimbProfile,
) -> float:
    image = np.asarray(image, dtype=np.float64)
    denominator = float(np.nansum(image))
    if denominator <= 0.0:
        raise ValueError("Solar image has no positive intensity")
    visible = terrain_limb_visible_map(
        shape=image.shape,
        center_column=center_column,
        center_row=center_row,
        pixels_per_radian=pixels_per_radian,
        geometry=geometry,
        profile=profile,
    )
    return float(np.nansum(image * visible) / denominator)
