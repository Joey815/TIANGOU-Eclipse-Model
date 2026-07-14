#!/usr/bin/env python3.11

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiangou.lunar_dem import discover_polar_ldem_pairs, iter_combined_lunar_point_blocks
from tiangou.lunar_limb import build_limb_profile_from_point_blocks
from tiangou.spice_geometry import compute_eclipse_geometry


def _build_native_profile_with_progress(
    geometry,
    args,
    lon_bounds,
    polar_sets,
):
    block_count = 0
    point_count = 0
    t0 = time.perf_counter()

    def _blocks():
        nonlocal block_count, point_count
        for block in iter_combined_lunar_point_blocks(
            global_img_path=args.img,
            global_lbl_path=args.lbl,
            lon_bounds_deg=lon_bounds,
            polar_datasets=polar_sets,
            global_row_block_size=args.global_row_block_size,
            global_col_block_size=args.global_col_block_size,
            polar_row_block_size=args.polar_row_block_size,
            polar_col_block_size=args.polar_col_block_size,
        ):
            block_count += 1
            point_count += int(block.shape[0])
            if block_count % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"      blocks={block_count}, points={point_count:,}, elapsed={elapsed:.1f}s")
            yield block

    profile = build_limb_profile_from_point_blocks(
        moon_point_blocks_xyz_km=_blocks(),
        observer_distance_km=geometry.moon_distance_km,
        sub_observer_lon_deg=geometry.sub_observer_lon_deg,
        sub_observer_lat_deg=geometry.sub_observer_lat_deg,
        n_bins=args.nbins,
    )
    return profile, block_count, point_count


def main() -> None:
    parser = ArgumentParser(description="Build a true-limb profile from an SLDEM IMG/LBL pair.")
    parser.add_argument("--time", required=True, help='UTC time, e.g. "2024-04-08T18:00:00"')
    parser.add_argument("--lon", required=True, type=float, help="Observer longitude in degrees east")
    parser.add_argument("--lat", required=True, type=float, help="Observer latitude in degrees north")
    parser.add_argument("--height-m", type=float, default=0.0, help="Orthometric height in meters")
    parser.add_argument("--geoid-height-m", type=float, default=0.0, help="Geoid height in meters")
    parser.add_argument("--img", required=True, help="Path to SLDEM IMG file")
    parser.add_argument("--lbl", required=True, help="Path to matching SLDEM LBL file")
    parser.add_argument("--lon-window-deg", type=float, default=120.0, help="Longitude swath centered on sub-observer longitude")
    parser.add_argument("--polar-dir", required=True, help="Directory containing the 60 N/S, 240 m polar LDEM pairs")
    parser.add_argument("--nbins", type=int, default=18000, help="Number of limb-profile bins")
    parser.add_argument("--global-row-block-size", type=int, default=128, help="In-memory row block size for SLDEM")
    parser.add_argument("--global-col-block-size", type=int, default=4096, help="In-memory column block size for SLDEM")
    parser.add_argument("--polar-row-block-size", type=int, default=128, help="In-memory row block size for polar LDEM")
    parser.add_argument("--polar-col-block-size", type=int, default=4096, help="In-memory column block size for polar LDEM")
    parser.add_argument("--kernel-dir", default=str(ROOT / "data" / "kernels"), help="Directory containing NAIF kernels")
    parser.add_argument("--output", required=True, help="Output .npz profile path")
    args = parser.parse_args()

    geometry = compute_eclipse_geometry(
        time_utc=args.time,
        lon_deg=args.lon,
        lat_deg=args.lat,
        height_m=args.height_m,
        geoid_height_m=args.geoid_height_m,
        kernel_dir=args.kernel_dir,
    )

    sub_lon = geometry.sub_observer_lon_deg % 360.0
    lon_bounds = (sub_lon - args.lon_window_deg, sub_lon + args.lon_window_deg)
    print(
        "[1/3] reading native SLDEM pixels, "
        f"sub-observer lon={sub_lon:.3f} deg, window={lon_bounds}"
    )
    polar_sets = discover_polar_ldem_pairs(polar_dir=args.polar_dir)
    print(f"      polar sets: {len(polar_sets)}")
    t0 = time.perf_counter()
    print(
        "[2/3] accumulating the native-resolution limb in memory-bounded blocks "
        f"(global={args.global_row_block_size}x{args.global_col_block_size}, "
        f"polar={args.polar_row_block_size}x{args.polar_col_block_size})"
    )
    profile, block_count, point_count = _build_native_profile_with_progress(
        geometry=geometry,
        args=args,
        lon_bounds=lon_bounds,
        polar_sets=polar_sets,
    )
    print(f"      read {point_count:,} lunar points across {block_count} blocks")
    elapsed_s = time.perf_counter() - t0

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        alpha_rad=profile.alpha_rad,
        theta_rad=profile.theta_rad,
        geometry=json.dumps(geometry.to_dict()),
        observer_distance_km=profile.observer_distance_km,
        sub_observer_lon_deg=profile.sub_observer_lon_deg,
        sub_observer_lat_deg=profile.sub_observer_lat_deg,
        elapsed_s=np.array(elapsed_s, dtype=np.float64),
        native_resolution=np.array(1, dtype=np.int8),
    )
    print(f"[3/3] wrote {output} in {elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
