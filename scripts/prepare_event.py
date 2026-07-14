#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def event_time(event: str, clock: str) -> str:
    datetime.strptime(f"{event}T{clock}", "%Y-%m-%dT%H:%M:%S")
    return f"{event}T{clock}Z"


def main() -> None:
    parser = ArgumentParser(description="Download and stage all external inputs for one TIANGOU event.")
    parser.add_argument("--event", required=True, help="YYYY-MM-DD")
    parser.add_argument("--start", default="15:00:00")
    parser.add_argument("--end", default="21:00:00")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--skip-aia", action="store_true")
    parser.add_argument("--skip-suvi", action="store_true")
    parser.add_argument("--skip-fism2", action="store_true")
    parser.add_argument("--skip-spice", action="store_true")
    parser.add_argument("--skip-dem", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    event_compact = args.event.replace("-", "")
    solar_root = args.data_root / "solar"
    start = event_time(args.event, args.start)
    end = event_time(args.event, args.end)

    if not args.skip_spice:
        run([python, str(ROOT / "scripts" / "download_spice_kernels.py"), "--kernel-dir", str(args.data_root / "kernels")])
    if not args.skip_dem:
        run([python, str(ROOT / "scripts" / "download_lunar_dem.py"), "--output-dir", str(args.data_root / "lunar_dem")])
    if not args.skip_fism2:
        run([python, str(ROOT / "scripts" / "download_fism2_event.py"), "--event", args.event, "--output-root", str(args.data_root / "fism2" / "by_date")])

    if not args.skip_aia:
        aia_root = solar_root / f"aia_{event_compact}_5min"
        run([python, str(ROOT / "scripts" / "download_aia_5min_archive.py"), "build-manifest", "--dataset-root", str(aia_root), "--start", start, "--end", end, "--step-minutes", "5"])
        run([python, str(ROOT / "scripts" / "download_aia_5min_archive.py"), "download", "--dataset-root", str(aia_root)])

    if not args.skip_suvi:
        suvi_root = solar_root / f"suvi_{event_compact}_5min"
        run([python, str(ROOT / "scripts" / "download_suvi_5min_archive.py"), "build-manifest", "--dataset-root", str(suvi_root), "--start", start, "--end", end, "--step-minutes", "5", "--bands", "ci284"])
        run([python, str(ROOT / "scripts" / "download_suvi_5min_archive.py"), "download", "--dataset-root", str(suvi_root), "--bands", "ci284"])


if __name__ == "__main__":
    main()
