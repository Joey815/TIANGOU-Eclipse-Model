#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Source:
    instrument: str
    wavelength: int
    directory_token: str
    image_subdir: str


SOURCES = (
    Source("aia", 94, "94", "94"),
    Source("aia", 131, "131", "131"),
    Source("aia", 171, "171", "171"),
    Source("aia", 193, "193", "193"),
    Source("aia", 211, "211", "211"),
    Source("suvi", 284, "suvi_ci284", "ci284"),
    Source("aia", 304, "304", "304"),
    Source("aia", 335, "335", "335"),
    Source("aia", 1600, "1600", "1600"),
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def parse_clock(event: str, clock: str) -> datetime:
    return datetime.strptime(f"{event}T{clock}", "%Y-%m-%dT%H:%M:%S")


def seconds_after_midnight(clock: str) -> int:
    value = datetime.strptime(clock, "%H:%M:%S")
    return value.hour * 3600 + value.minute * 60 + value.second


def build_times(start: datetime, end: datetime, step_minutes: int) -> list[datetime]:
    if end < start:
        raise ValueError("End time must not precede start time")
    values: list[datetime] = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(minutes=step_minutes)
    return values


def load_heights(path: Path) -> list[float]:
    return [
        float(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def source_image_dir(solar_root: Path, event_compact: str, source: Source) -> Path:
    family = "aia" if source.instrument == "aia" else "suvi"
    return solar_root / f"{family}_{event_compact}_5min" / "files" / source.image_subdir


def main() -> None:
    parser = ArgumentParser(description="Run the native-resolution TIANGOU workflow on a local computer.")
    parser.add_argument("--event", required=True, help="YYYY-MM-DD")
    parser.add_argument("--start", default="15:00:00")
    parser.add_argument("--end", default="21:00:00")
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--solar-root", type=Path)
    parser.add_argument("--fism-root", type=Path)
    parser.add_argument("--dem-root", type=Path)
    parser.add_argument("--kernel-root", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="Local worker processes used by each snapshot")
    parser.add_argument("--dry-run", action="store_true", help="Check the plan without calculating masks")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.step_minutes < 1:
        parser.error("--step-minutes must be at least 1")

    event_compact = args.event.replace("-", "")
    start = parse_clock(args.event, args.start)
    end = parse_clock(args.event, args.end)
    times = build_times(start, end, args.step_minutes)
    grid_file = ROOT / "pipeline" / "vertical_grids" / "waccm6_70_heights_km.txt"
    heights = load_heights(grid_file)

    solar_root = (args.solar_root or args.data_root / "solar").resolve()
    fism_root = (args.fism_root or args.data_root / "fism2" / "by_date").resolve()
    dem_root = (args.dem_root or args.data_root / "lunar_dem").resolve()
    kernel_root = (args.kernel_root or args.data_root / "kernels").resolve()
    mask_root = (args.data_root / "output" / "source_masks").resolve()
    final_root = (args.data_root / "output" / "final_masks").resolve()
    shared_profiles = mask_root / f"{event_compact}_tiangou_lunar_profiles"

    final_dir = final_root / f"{event_compact}_tiangou_23band"
    final_nc = final_dir / f"TIANGOU_Mask_{args.event.replace('-', '_')}_waccm6_70_23band.nc"

    print(f"event={args.event}")
    print(f"times={len(times)}")
    print(f"heights={len(heights)}")
    print(f"sources={len(SOURCES)}")
    print(f"snapshots={len(times) * len(heights) * len(SOURCES)}")
    print(f"workers={args.workers}")
    print(f"final_nc={final_nc}")
    if args.dry_run:
        return

    if not args.skip_preflight:
        preflight = [
            sys.executable,
            str(ROOT / "scripts" / "preflight.py"),
            "--event",
            args.event,
            "--data-root",
            str(args.data_root),
            "--solar-root",
            str(solar_root),
            "--fism-root",
            str(fism_root),
            "--dem-root",
            str(dem_root),
            "--kernel-root",
            str(kernel_root),
        ]
        run(preflight)

    if final_nc.is_file():
        run([sys.executable, str(ROOT / "scripts" / "validate_eclipsemask.py"), str(final_nc), "--expected-times", str(len(times))])
        return

    total_per_source = len(times) * len(heights)
    for source_index, source in enumerate(SOURCES, start=1):
        run_tag = f"{event_compact}_{source.directory_token}_5min_waccm6_70_tiangou"
        run_root = mask_root / run_tag
        raw_dir = run_root / "raw_nc"
        source_nc = run_root / "TIANGOU_SourceMask.nc"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shared_profiles.mkdir(parents=True, exist_ok=True)
        image_dir = source_image_dir(solar_root, event_compact, source)

        completed = sum(1 for _ in raw_dir.glob("*.nc"))
        print(
            f"source={source_index}/{len(SOURCES)} id={source.directory_token} "
            f"completed={completed}/{total_per_source}"
        )
        for time_index, time_value in enumerate(times, start=1):
            stamp = time_value.strftime("%Y%m%d%H%M%S")
            for height_index, height_km in enumerate(heights, start=1):
                output_nc = raw_dir / f"{stamp}_{height_km:.2f}km_{source.wavelength}.nc"
                if output_nc.is_file():
                    continue
                print(
                    f"  time={time_index}/{len(times)} height={height_index}/{len(heights)} "
                    f"utc={time_value:%Y-%m-%dT%H:%M:%S} z={height_km:.2f}km"
                )
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "compute_tiangou_snapshot.py"),
                    "--time",
                    time_value.strftime("%Y-%m-%dT%H:%M:%S"),
                    "--height-km",
                    f"{height_km:.2f}",
                    "--output",
                    str(output_nc),
                    "--profile-dir",
                    str(shared_profiles),
                    "--image-folder",
                    str(image_dir),
                    "--instrument",
                    source.instrument,
                    "--wl",
                    str(source.wavelength),
                    "--workers",
                    str(args.workers),
                    "--kernel-dir",
                    str(kernel_root),
                    "--img",
                    str(dem_root / "sldem2015_128_60s_60n_000_360_float.img"),
                    "--lbl",
                    str(dem_root / "sldem2015_128_60s_60n_000_360_float.lbl"),
                    "--polar-dir",
                    str(dem_root / "polar"),
                    "--metadata-json",
                    str(output_nc.with_suffix(".json")),
                ]
                run(command)

        run(
            [
                sys.executable,
                str(ROOT / "pipeline" / "assemble_source_cube.py"),
                str(raw_dir),
                "--output",
                str(source_nc),
            ]
        )

    final_dir.mkdir(parents=True, exist_ok=True)
    start_sec = seconds_after_midnight(args.start)
    end_sec = seconds_after_midnight(args.end)
    run(
        [
            sys.executable,
            str(ROOT / "pipeline" / "make_tiangou_23band_mask.py"),
            "--event",
            event_compact,
            "--mask-root",
            str(mask_root),
            "--fism-root",
            str(fism_root),
            "--z-file",
            str(grid_file),
            "--cross-section-file",
            str(ROOT / "pipeline" / "stan_bands" / "cross_sections.dat"),
            "--start-sec",
            str(start_sec),
            "--end-sec",
            str(end_sec),
            "--step-sec",
            str(args.step_minutes * 60),
            "--source-start-sec",
            str(start_sec),
            "--output",
            str(final_nc),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_eclipsemask.py"),
            str(final_nc),
            "--expected-times",
            str(len(times)),
            "--json-output",
            str(final_nc.with_suffix(".validation.json")),
        ]
    )


if __name__ == "__main__":
    main()
