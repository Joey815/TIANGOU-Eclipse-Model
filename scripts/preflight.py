#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AIA_BANDS = ("94", "131", "171", "193", "211", "304", "335", "1600")
DEM_SIZES = {
    "sldem2015_128_60s_60n_000_360_float.img": 2_831_155_200,
    "sldem2015_128_60s_60n_000_360_float.lbl": 4_588,
    "polar/ldem_60n_240m_float.img": 240_870_400,
    "polar/ldem_60n_240m_float.lbl": 4_809,
    "polar/ldem_60s_240m_float.img": 240_870_400,
    "polar/ldem_60s_240m_float.lbl": 4_877,
}
KERNELS = (
    "naif0012.tls",
    "pck00011.tpc",
    "de440.bsp",
    "earth_latest_high_prec.bpc",
    "moon_pa_de440_200625.bpc",
    "moon_de440_200625.tf",
)


def add_check(checks: list[dict], label: str, path: Path, ok: bool, detail: str = "") -> None:
    checks.append({"label": label, "path": str(path), "ok": bool(ok), "detail": detail})


def fits_count(directory: Path) -> int:
    return sum(1 for pattern in ("*.fits", "*.fit", "*.fts") for _ in directory.glob(pattern))


def main() -> None:
    parser = ArgumentParser(description="Verify every input required by the TIANGOU nine-source workflow.")
    parser.add_argument("--event", required=True, help="YYYY-MM-DD")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--solar-root", type=Path)
    parser.add_argument("--fism-root", type=Path)
    parser.add_argument("--dem-root", type=Path)
    parser.add_argument("--kernel-root", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    event_compact = args.event.replace("-", "")
    solar_root = args.solar_root or args.data_root / "solar"
    fism_root = args.fism_root or args.data_root / "fism2" / "by_date"
    dem_root = args.dem_root or args.data_root / "lunar_dem"
    kernel_root = args.kernel_root or args.data_root / "kernels"
    checks: list[dict] = []

    aia_root = solar_root / f"aia_{event_compact}_5min" / "files"
    for band in AIA_BANDS:
        directory = aia_root / band
        count = fits_count(directory) if directory.is_dir() else 0
        add_check(checks, f"AIA {band}", directory, count > 0, f"fits={count}")

    suvi_dir = solar_root / f"suvi_{event_compact}_5min" / "files" / "ci284"
    suvi_count = fits_count(suvi_dir) if suvi_dir.is_dir() else 0
    add_check(checks, "SUVI 284", suvi_dir, suvi_count > 0, f"fits={suvi_count}")

    event_fism = fism_root / args.event
    for label, pattern in (
        ("FISM2 flare_hr", "flare_hr/netcdf/FISM_60sec_*.nc"),
        ("FISM2 flare_bands", "flare_bands/netcdf/FISM_bands_*.nc"),
    ):
        matches = sorted(event_fism.glob(pattern))
        add_check(checks, label, event_fism / pattern, len(matches) == 1, f"matches={len(matches)}")

    for relative, expected_size in DEM_SIZES.items():
        path = dem_root / relative
        size = path.stat().st_size if path.exists() else -1
        add_check(checks, f"DEM {relative}", path, size == expected_size, f"size={size}, expected={expected_size}")

    for name in KERNELS:
        path = kernel_root / name
        size = path.stat().st_size if path.exists() else 0
        add_check(checks, f"SPICE {name}", path, size > 0, f"size={size}")

    ok = all(item["ok"] for item in checks)
    report = {"event": args.event, "ok": ok, "checks": checks}
    for item in checks:
        status = "OK" if item["ok"] else "MISSING"
        print(f"{status:7s} {item['label']}: {item['path']} {item['detail']}")
    print(f"preflight_ok={str(ok).lower()}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
