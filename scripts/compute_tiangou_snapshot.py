#!/usr/bin/env python3.11

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from argparse import ArgumentParser

import numpy as np
import xarray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiangou.image_occultor import terrain_limb_image_transmission
from tiangou.lunar_dem import discover_polar_ldem_pairs, iter_combined_lunar_point_blocks
from tiangou.lunar_limb import LimbProfile, build_limb_profile_from_point_blocks
from tiangou.profile_io import load_limb_profile_npz
from tiangou.solar_image import load_solar_image
from tiangou.spice_geometry import compute_eclipse_geometry, compute_grid_geometry


MODEL_NAME = "TIANGOU Eclipse Model"
MODEL_FULL_NAME = "Topography-aware Irradiance And Nonuniform-Geometry Occultation Utility"
_WORKER: dict[str, object] = {}


def _write_netcdf_atomic(
    dataset: xarray.Dataset,
    output: str | Path,
    retries: int = 4,
    base_sleep: float = 2.0,
) -> None:
    output = os.path.abspath(str(output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    for attempt in range(1, retries + 1):
        temporary = f"{output}.tmp.{os.getpid()}.{attempt}"
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
            dataset.to_netcdf(temporary)
            os.replace(temporary, output)
            return
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            if attempt == retries:
                raise
            time.sleep(base_sleep * attempt)


def _save_profile_npz(profile: LimbProfile, geometry, output: Path, elapsed_s: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        alpha_rad=profile.alpha_rad,
        theta_rad=profile.theta_rad,
        geometry=json.dumps(asdict(geometry)),
        observer_distance_km=profile.observer_distance_km,
        sub_observer_lon_deg=profile.sub_observer_lon_deg,
        sub_observer_lat_deg=profile.sub_observer_lat_deg,
        elapsed_s=np.array(elapsed_s, dtype=np.float64),
        native_resolution=np.array(1, dtype=np.int8),
    )


def _init_worker(
    image: np.ndarray,
    center_column: float,
    center_row: float,
    pixels_per_radian: float,
    profile_alpha: np.ndarray,
    profile_theta: np.ndarray,
    profile_observer_distance_km: float,
    profile_sub_lon_deg: float,
    profile_sub_lat_deg: float,
    time_utc: str,
    height_m: float,
    kernel_dir: str,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    _WORKER.update(
        image=image,
        center_column=float(center_column),
        center_row=float(center_row),
        pixels_per_radian=float(pixels_per_radian),
        profile=LimbProfile(
            alpha_rad=np.asarray(profile_alpha, dtype=float),
            theta_rad=np.asarray(profile_theta, dtype=float),
            observer_distance_km=float(profile_observer_distance_km),
            sub_observer_lon_deg=float(profile_sub_lon_deg),
            sub_observer_lat_deg=float(profile_sub_lat_deg),
        ),
        time_utc=time_utc,
        height_m=float(height_m),
        kernel_dir=kernel_dir,
    )


def _compute_point(task: tuple[int, int, float, float]) -> tuple[int, int, float]:
    row, column, lon, lat = task
    geometry = compute_eclipse_geometry(
        time_utc=_WORKER["time_utc"],
        lon_deg=lon,
        lat_deg=lat,
        height_m=_WORKER["height_m"],
        kernel_dir=_WORKER["kernel_dir"],
    )
    value = terrain_limb_image_transmission(
        image=_WORKER["image"],
        center_column=_WORKER["center_column"],
        center_row=_WORKER["center_row"],
        pixels_per_radian=_WORKER["pixels_per_radian"],
        geometry=geometry,
        profile=_WORKER["profile"],
    )
    return row, column, value


def _build_or_load_profile(
    profile_path: Path,
    time_utc: str,
    lon_deg: float,
    lat_deg: float,
    height_m: float,
    args,
):
    if profile_path.exists():
        cached = load_limb_profile_npz(profile_path)
        if int(cached.metadata.get("native_resolution", 0)) != 1:
            raise ValueError(f"Lunar profile is not native-resolution data: {profile_path}")
        return cached

    geometry = compute_eclipse_geometry(
        time_utc=time_utc,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        height_m=height_m,
        kernel_dir=args.kernel_dir,
    )
    sub_lon = geometry.sub_observer_lon_deg % 360.0
    lon_bounds = (sub_lon - args.lon_window_deg, sub_lon + args.lon_window_deg)
    polar_sets = discover_polar_ldem_pairs(polar_dir=args.polar_dir)

    started = time.perf_counter()
    profile = build_limb_profile_from_point_blocks(
        moon_point_blocks_xyz_km=iter_combined_lunar_point_blocks(
            global_img_path=args.img,
            global_lbl_path=args.lbl,
            lon_bounds_deg=lon_bounds,
            polar_datasets=polar_sets,
            global_row_block_size=args.global_row_block_size,
            global_col_block_size=args.global_col_block_size,
            polar_row_block_size=args.polar_row_block_size,
            polar_col_block_size=args.polar_col_block_size,
        ),
        observer_distance_km=geometry.moon_distance_km,
        sub_observer_lon_deg=geometry.sub_observer_lon_deg,
        sub_observer_lat_deg=geometry.sub_observer_lat_deg,
        n_bins=args.nbins,
    )
    elapsed_s = time.perf_counter() - started
    _save_profile_npz(profile, geometry, profile_path, elapsed_s)
    return load_limb_profile_npz(profile_path)


def _image_field_radius(
    shape: tuple[int, int],
    center_column: float,
    center_row: float,
    pixels_per_radian: float,
) -> float:
    corner_rows = np.array([0.0, 0.0, shape[0] - 1.0, shape[0] - 1.0])
    corner_columns = np.array([0.0, shape[1] - 1.0, 0.0, shape[1] - 1.0])
    return float(
        np.max(np.hypot(corner_rows - center_row, corner_columns - center_column))
        / pixels_per_radian
    )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Generate one TIANGOU source-mask snapshot.")
    parser.add_argument("--time", required=True, help='UTC time, e.g. "2024-04-08T18:32:00"')
    parser.add_argument("--height-km", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--instrument", required=True, choices=["aia", "suvi"])
    parser.add_argument("--wl", required=True, type=int)
    parser.add_argument("--sn", default=16, type=int)
    parser.add_argument("--glon-min", type=float, default=-180.0)
    parser.add_argument("--glon-max", type=float, default=180.0)
    parser.add_argument("--glat-min", type=float, default=-90.0)
    parser.add_argument("--glat-max", type=float, default=90.0)
    parser.add_argument("--dlon", type=float, default=1.25)
    parser.add_argument("--dlat", type=float, default=0.9375)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--kernel-dir", default=str(ROOT / "data" / "kernels"))
    parser.add_argument("--img", default=str(ROOT / "data" / "lunar_dem" / "sldem2015_128_60s_60n_000_360_float.img"))
    parser.add_argument("--lbl", default=str(ROOT / "data" / "lunar_dem" / "sldem2015_128_60s_60n_000_360_float.lbl"))
    parser.add_argument("--lon-window-deg", type=float, default=120.0)
    parser.add_argument("--polar-dir", default=str(ROOT / "data" / "lunar_dem" / "polar"))
    parser.add_argument("--nbins", type=int, default=18000)
    parser.add_argument("--global-row-block-size", type=int, default=128)
    parser.add_argument("--global-col-block-size", type=int, default=4096)
    parser.add_argument("--polar-row-block-size", type=int, default=128)
    parser.add_argument("--polar-col-block-size", type=int, default=4096)
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    output = Path(args.output).expanduser().resolve()
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    metadata_json = (
        Path(args.metadata_json).expanduser().resolve()
        if args.metadata_json
        else output.with_suffix(".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        print(f"skip_existing={output}")
        return

    started = time.perf_counter()
    time_value = datetime.strptime(args.time, "%Y-%m-%dT%H:%M:%S")
    solar = load_solar_image(
        folder=args.image_folder,
        instrument=args.instrument,
        wavelength=args.wl,
        target_time=time_value,
        satellite_number=args.sn,
    )
    glon = np.arange(args.glon_min, args.glon_max, args.dlon, dtype=np.float32)
    glat = np.arange(args.glat_min, args.glat_max, args.dlat, dtype=np.float32)

    grid = compute_grid_geometry(
        time_utc=time_value,
        lon_deg=glon,
        lat_deg=glat,
        height_m=args.height_km * 1000.0,
        kernel_dir=args.kernel_dir,
    )
    field_radius = _image_field_radius(
        solar.data.shape,
        solar.center_column,
        solar.center_row,
        solar.pixels_per_radian,
    )
    candidate = (
        (grid.solar_zenith_angle_rad < np.pi / 2.0)
        & (grid.sun_moon_separation_rad <= field_radius + grid.moon_angular_radius_rad)
    )
    candidate_count = int(np.count_nonzero(candidate))
    transmission = np.ones((glat.size, glon.size), dtype=np.float32)
    profile_path: Path | None = None
    profile_elapsed = 0.0
    integration_elapsed = 0.0

    if candidate_count:
        distances = np.where(candidate, grid.sun_moon_separation_rad, np.inf)
        profile_row, profile_column = np.unravel_index(int(np.argmin(distances)), distances.shape)
        profile_lon = float(glon[profile_column])
        profile_lat = float(glat[profile_row])
        profile_path = profile_dir / (
            f"profile_{time_value:%Y%m%dT%H%M%S}_{args.height_km:06.2f}km.npz"
        )

        profile_started = time.perf_counter()
        cached = _build_or_load_profile(
            profile_path=profile_path,
            time_utc=args.time,
            lon_deg=profile_lon,
            lat_deg=profile_lat,
            height_m=args.height_km * 1000.0,
            args=args,
        )
        profile_elapsed = time.perf_counter() - profile_started
        tasks = [
            (int(row), int(column), float(glon[column]), float(glat[row]))
            for row, column in np.column_stack(np.where(candidate))
        ]
        worker_args = (
            solar.data,
            solar.center_column,
            solar.center_row,
            solar.pixels_per_radian,
            cached.profile.alpha_rad,
            cached.profile.theta_rad,
            cached.profile.observer_distance_km,
            cached.profile.sub_observer_lon_deg,
            cached.profile.sub_observer_lat_deg,
            args.time,
            args.height_km * 1000.0,
            args.kernel_dir,
        )

        integration_started = time.perf_counter()
        if args.workers == 1:
            _init_worker(*worker_args)
            results = map(_compute_point, tasks)
            for row, column, value in results:
                transmission[row, column] = value
        else:
            chunksize = max(1, len(tasks) // (args.workers * 8))
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=worker_args,
            ) as executor:
                for row, column, value in executor.map(
                    _compute_point,
                    tasks,
                    chunksize=chunksize,
                ):
                    transmission[row, column] = value
        integration_elapsed = time.perf_counter() - integration_started

    dataset = xarray.Dataset(
        {"transmission": (("glat", "glon"), transmission)},
        coords={"glon": glon, "glat": glat},
    )
    dataset["time"] = np.datetime64(time_value)
    dataset["alt_km"] = np.float32(args.height_km)
    dataset["wavelength_angstrom"] = np.int32(args.wl)
    dataset["image_time"] = np.datetime64(solar.observation_time)
    dataset["instrument"] = args.instrument
    dataset.attrs["model_name"] = MODEL_NAME
    dataset.attrs["model_full_name"] = MODEL_FULL_NAME
    dataset.attrs["native_resolution"] = "solar image and lunar DEM pixels used without skipping"
    dataset.attrs["solar_image_rows"] = int(solar.data.shape[0])
    dataset.attrs["solar_image_columns"] = int(solar.data.shape[1])
    dataset.attrs["solar_image_file"] = str(solar.source_path)
    dataset.attrs["lunar_profile_file"] = str(profile_path) if profile_path else ""
    dataset.attrs["kernel_dir"] = str(Path(args.kernel_dir).expanduser().resolve())
    _write_netcdf_atomic(dataset, output)
    dataset.close()

    total_elapsed = time.perf_counter() - started
    metadata = {
        "time_utc": args.time,
        "height_km": args.height_km,
        "instrument": args.instrument,
        "wavelength_angstrom": args.wl,
        "solar_image_file": str(solar.source_path),
        "solar_image_shape": list(solar.data.shape),
        "output_nc": str(output),
        "candidate_count": candidate_count,
        "transmission_min": float(np.nanmin(transmission)),
        "transmission_max": float(np.nanmax(transmission)),
        "elapsed_total_s": total_elapsed,
        "elapsed_profile_s": profile_elapsed,
        "elapsed_integration_s": integration_elapsed,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
