#!/usr/bin/env python3.11

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiangou.spice_geometry import compute_eclipse_geometry


def main() -> None:
    parser = ArgumentParser(description="Inspect SPICE-based eclipse geometry at one observer/time.")
    parser.add_argument("--time", required=True, help='UTC time, e.g. "2024-04-08T18:00:00"')
    parser.add_argument("--lon", required=True, type=float, help="Observer longitude in degrees east")
    parser.add_argument("--lat", required=True, type=float, help="Observer latitude in degrees north")
    parser.add_argument("--height-m", type=float, default=0.0, help="Orthometric height in meters")
    parser.add_argument("--geoid-height-m", type=float, default=0.0, help="Geoid height in meters")
    parser.add_argument(
        "--kernel-dir",
        default=str(ROOT / "data" / "kernels"),
        help="Directory containing NAIF kernels",
    )
    args = parser.parse_args()

    geometry = compute_eclipse_geometry(
        time_utc=args.time,
        lon_deg=args.lon,
        lat_deg=args.lat,
        height_m=args.height_m,
        geoid_height_m=args.geoid_height_m,
        kernel_dir=args.kernel_dir,
    )
    print(json.dumps(geometry.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
