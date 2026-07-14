#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path

from download_utils import download_resumable


FISM_BASE = "https://lasp.colorado.edu/eve/data_access/evewebdata/fism"


def event_tokens(event: str) -> tuple[datetime, str]:
    event_date = datetime.strptime(event, "%Y-%m-%d")
    return event_date, event_date.strftime("%Y%j")


def product_records(event: str, output_root: Path) -> list[dict[str, str]]:
    event_date, ydoy = event_tokens(event)
    year = event_date.strftime("%Y")
    return [
        {
            "product": "flare_hr",
            "url": f"{FISM_BASE}/flare_hr_data/netcdf/{year}/FISM_60sec_{ydoy}_v02_01.nc",
            "path": str(output_root / event / "flare_hr" / "netcdf" / f"FISM_60sec_{ydoy}_v02_01.nc"),
        },
        {
            "product": "flare_bands",
            "url": f"{FISM_BASE}/flare_bands/netcdf/{year}/FISM_bands_{ydoy}_v02_01.nc",
            "path": str(output_root / event / "flare_bands" / "netcdf" / f"FISM_bands_{ydoy}_v02_01.nc"),
        },
    ]


def main() -> None:
    parser = ArgumentParser(description="Download official FISM2 flare products for one event date.")
    parser.add_argument("--event", required=True, help="Event date in YYYY-MM-DD format")
    parser.add_argument("--output-root", type=Path, required=True, help="Target by_date directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    records = product_records(args.event, args.output_root)
    for record in records:
        path = download_resumable(record["url"], record["path"], overwrite=args.overwrite)
        record["size_bytes"] = str(path.stat().st_size)

    event_root = args.output_root / args.event
    event_root.mkdir(parents=True, exist_ok=True)
    (event_root / "download_manifest.json").write_text(
        json.dumps({"event": args.event, "products": records}, indent=2) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
