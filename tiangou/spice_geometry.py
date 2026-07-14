from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy as spice


EARTH_EQUATORIAL_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563
MOON_DATUM_RADIUS_KM = 1737.4
SUN_RADIUS_KM = 695700.0
_LOADED_KERNEL_SIGNATURE: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EclipseGeometry:
    utc: str
    et: float
    observer_lon_deg: float
    observer_lat_deg: float
    ellipsoid_height_m: float
    geoid_height_m: float
    moon_distance_km: float
    sun_distance_km: float
    sub_observer_lon_deg: float
    sub_observer_lat_deg: float
    sun_moon_separation_rad: float
    sun_position_angle_rad: float
    moon_axis_angle_rad: float
    sun_angular_radius_rad: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _normalize_time(value: str | datetime) -> str:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    return value


@dataclass(frozen=True)
class GridGeometry:
    solar_zenith_angle_rad: np.ndarray
    sun_moon_separation_rad: np.ndarray
    moon_angular_radius_rad: np.ndarray


def load_kernels(kernel_dir: str | Path, force_reload: bool = False) -> list[Path]:
    global _LOADED_KERNEL_SIGNATURE
    kernel_dir = Path(kernel_dir).expanduser().resolve()
    names = [
        "naif0012.tls",
        "pck00011.tpc",
        "earth_latest_high_prec.bpc",
        "moon_pa_de440_200625.bpc",
        "moon_de440_200625.tf",
        "de440.bsp",
    ]
    paths = [kernel_dir / name for name in names]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing kernels: {missing}")

    signature = tuple(str(path) for path in paths)
    if force_reload or _LOADED_KERNEL_SIGNATURE != signature:
        spice.kclear()
        for path in paths:
            spice.furnsh(str(path))
        _LOADED_KERNEL_SIGNATURE = signature
    return paths


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    return float(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))


def _sky_basis(target_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    icrf_north = np.array([0.0, 0.0, 1.0], dtype=float)
    north = icrf_north - np.dot(icrf_north, target_unit) * target_unit
    north_norm = np.linalg.norm(north)
    if north_norm < 1e-12:
        north = np.array([0.0, 1.0, 0.0], dtype=float)
        north = north - np.dot(north, target_unit) * target_unit
        north_norm = np.linalg.norm(north)
    north /= north_norm
    east = np.cross(north, target_unit)
    east /= np.linalg.norm(east)
    return north, east


def _position_angle(projected_vec: np.ndarray, target_unit: np.ndarray) -> float:
    north, east = _sky_basis(target_unit)
    sky_vec = projected_vec - np.dot(projected_vec, target_unit) * target_unit
    sky_norm = np.linalg.norm(sky_vec)
    if sky_norm < 1e-12:
        return 0.0
    sky_vec /= sky_norm
    return float(np.mod(np.arctan2(np.dot(sky_vec, east), np.dot(sky_vec, north)), 2.0 * np.pi))


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Zero-length vector cannot be normalized.")
    return vec / norm


def observer_icrf_vector(
    et: float,
    lon_deg: float,
    lat_deg: float,
    height_m: float,
    geoid_height_m: float = 0.0,
) -> np.ndarray:
    ellipsoid_height_km = (height_m + geoid_height_m) / 1000.0
    obs_itrf = spice.georec(
        np.deg2rad(lon_deg),
        np.deg2rad(lat_deg),
        ellipsoid_height_km,
        EARTH_EQUATORIAL_RADIUS_KM,
        EARTH_FLATTENING,
    )
    omat = spice.pxform("ITRF93", "J2000", et)
    return np.asarray(spice.mxv(omat, obs_itrf), dtype=float)


def compute_eclipse_geometry(
    time_utc: str | datetime,
    lon_deg: float,
    lat_deg: float,
    height_m: float = 0.0,
    geoid_height_m: float = 0.0,
    kernel_dir: str | Path = "data/kernels",
) -> EclipseGeometry:
    load_kernels(kernel_dir=kernel_dir)

    utc = _normalize_time(time_utc)
    et = float(spice.utc2et(utc))
    obs_icrf = observer_icrf_vector(
        et=et,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        height_m=height_m,
        geoid_height_m=geoid_height_m,
    )

    moon_geocentric = np.asarray(spice.spkezp(301, et, "J2000", "LT+S", 399)[0], dtype=float)
    sun_geocentric = np.asarray(spice.spkezp(10, et, "J2000", "LT+S", 399)[0], dtype=float)

    moon_vec = moon_geocentric - obs_icrf
    sun_vec = sun_geocentric - obs_icrf
    moon_distance_km = float(np.linalg.norm(moon_vec))
    sun_distance_km = float(np.linalg.norm(sun_vec))
    moon_unit = moon_vec / moon_distance_km

    sun_moon_separation_rad = _angle_between(sun_vec, moon_vec)
    sun_position_angle_rad = _position_angle(projected_vec=sun_vec, target_unit=moon_unit)

    moon_to_obs_icrf = obs_icrf - moon_geocentric
    mmat = spice.pxform("J2000", "MOON_ME_DE440_ME421", et)
    moon_to_obs_me = np.asarray(spice.mxv(mmat, moon_to_obs_icrf), dtype=float)
    dist, lam, bet = spice.reclat(moon_to_obs_me)
    _ = dist

    moon_frame_to_icrf = spice.pxform("MOON_ME_DE440_ME421", "J2000", et)
    moon_north_icrf = np.asarray(spice.mxv(moon_frame_to_icrf, [0.0, 0.0, 1.0]), dtype=float)
    moon_axis_angle_rad = _position_angle(projected_vec=moon_north_icrf, target_unit=moon_unit)

    return EclipseGeometry(
        utc=utc,
        et=et,
        observer_lon_deg=float(lon_deg),
        observer_lat_deg=float(lat_deg),
        ellipsoid_height_m=float(height_m + geoid_height_m),
        geoid_height_m=float(geoid_height_m),
        moon_distance_km=moon_distance_km,
        sun_distance_km=sun_distance_km,
        sub_observer_lon_deg=float(np.rad2deg(lam)),
        sub_observer_lat_deg=float(np.rad2deg(bet)),
        sun_moon_separation_rad=sun_moon_separation_rad,
        sun_position_angle_rad=sun_position_angle_rad,
        moon_axis_angle_rad=moon_axis_angle_rad,
        sun_angular_radius_rad=float(np.arcsin(SUN_RADIUS_KM / sun_distance_km)),
    )


def _geodetic_grid_itrf(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    lon_grid, lat_grid = np.meshgrid(
        np.deg2rad(np.asarray(lon_deg, dtype=np.float64)),
        np.deg2rad(np.asarray(lat_deg, dtype=np.float64)),
    )
    height_km = float(height_m) / 1000.0
    eccentricity_squared = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
    sin_lat = np.sin(lat_grid)
    cos_lat = np.cos(lat_grid)
    prime_vertical = EARTH_EQUATORIAL_RADIUS_KM / np.sqrt(
        1.0 - eccentricity_squared * sin_lat**2
    )

    positions = np.stack(
        (
            (prime_vertical + height_km) * cos_lat * np.cos(lon_grid),
            (prime_vertical + height_km) * cos_lat * np.sin(lon_grid),
            (prime_vertical * (1.0 - eccentricity_squared) + height_km) * sin_lat,
        ),
        axis=-1,
    )
    zenith = np.stack(
        (
            cos_lat * np.cos(lon_grid),
            cos_lat * np.sin(lon_grid),
            sin_lat,
        ),
        axis=-1,
    )
    return positions, zenith


def compute_grid_geometry(
    time_utc: str | datetime,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    height_m: float = 0.0,
    kernel_dir: str | Path = "data/kernels",
) -> GridGeometry:
    load_kernels(kernel_dir=kernel_dir)
    utc = _normalize_time(time_utc)
    et = float(spice.utc2et(utc))

    observer_itrf, zenith_itrf = _geodetic_grid_itrf(lon_deg, lat_deg, height_m)
    itrf_to_j2000 = np.asarray(spice.pxform("ITRF93", "J2000", et), dtype=np.float64)
    observer_j2000 = observer_itrf @ itrf_to_j2000.T
    zenith_j2000 = zenith_itrf @ itrf_to_j2000.T

    moon_geocentric = np.asarray(spice.spkezp(301, et, "J2000", "LT+S", 399)[0], dtype=np.float64)
    sun_geocentric = np.asarray(spice.spkezp(10, et, "J2000", "LT+S", 399)[0], dtype=np.float64)
    moon_vector = moon_geocentric - observer_j2000
    sun_vector = sun_geocentric - observer_j2000
    moon_distance = np.linalg.norm(moon_vector, axis=-1)
    sun_distance = np.linalg.norm(sun_vector, axis=-1)

    moon_unit = moon_vector / moon_distance[..., None]
    sun_unit = sun_vector / sun_distance[..., None]
    separation = np.arccos(np.clip(np.sum(moon_unit * sun_unit, axis=-1), -1.0, 1.0))
    solar_zenith = np.arccos(np.clip(np.sum(sun_unit * zenith_j2000, axis=-1), -1.0, 1.0))
    moon_radius = np.arcsin(np.clip(MOON_DATUM_RADIUS_KM / moon_distance, 0.0, 1.0))

    return GridGeometry(
        solar_zenith_angle_rad=solar_zenith,
        sun_moon_separation_rad=separation,
        moon_angular_radius_rad=moon_radius,
    )


def compute_moon_observer_basis_me(
    time_utc: str | datetime,
    lon_deg: float,
    lat_deg: float,
    height_m: float = 0.0,
    geoid_height_m: float = 0.0,
    kernel_dir: str | Path = "data/kernels",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    load_kernels(kernel_dir=kernel_dir)

    utc = _normalize_time(time_utc)
    et = float(spice.utc2et(utc))
    obs_icrf = observer_icrf_vector(
        et=et,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        height_m=height_m,
        geoid_height_m=geoid_height_m,
    )

    moon_geocentric = np.asarray(spice.spkezp(301, et, "J2000", "LT+S", 399)[0], dtype=float)
    moon_from_observer_icrf = moon_geocentric - obs_icrf
    moon_to_observer_icrf = obs_icrf - moon_geocentric

    moon_unit_icrf = _normalize(moon_from_observer_icrf)
    sky_north_icrf, sky_east_icrf = _sky_basis(moon_unit_icrf)

    j2000_to_moon = spice.pxform("J2000", "MOON_ME_DE440_ME421", et)
    observer_dir_me = _normalize(np.asarray(spice.mxv(j2000_to_moon, moon_to_observer_icrf), dtype=float))
    sky_north_me = _normalize(np.asarray(spice.mxv(j2000_to_moon, sky_north_icrf), dtype=float))
    sky_east_me = _normalize(np.asarray(spice.mxv(j2000_to_moon, sky_east_icrf), dtype=float))

    sky_east_me = _normalize(sky_east_me - np.dot(sky_east_me, observer_dir_me) * observer_dir_me)
    sky_north_me = _normalize(sky_north_me - np.dot(sky_north_me, observer_dir_me) * observer_dir_me)
    sky_north_me = _normalize(sky_north_me - np.dot(sky_north_me, sky_east_me) * sky_east_me)

    return observer_dir_me, sky_north_me, sky_east_me
