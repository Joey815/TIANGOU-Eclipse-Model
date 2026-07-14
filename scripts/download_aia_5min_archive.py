from __future__ import annotations

from argparse import ArgumentParser
from bisect import bisect_left
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

from astropy.io import fits
from dateutil import parser
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "data" / "solar" / "aia_20231014_5min"
BANDS = ["94", "131", "171", "193", "211", "304", "335", "1600"]
JSOC_MIRRORS = ["http://jsoc.stanford.edu", "http://jsoc2.stanford.edu"]
EUV_SERIES = "aia.lev1_euv_12s"
UV_SERIES = "aia.lev1_uv_24s"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class Layout:
    root: Path
    files_dir: Path
    manifest_dir: Path
    logs_dir: Path
    status_dir: Path
    manifest_csv: Path
    manifest_json: Path
    readme_txt: Path
    log_jsonl: Path
    summary_json: Path


def parse_utc(value: str) -> datetime:
    dt = parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def format_utc(value: datetime) -> str:
    return value.strftime(TIME_FORMAT)


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def build_layout(root: Path) -> Layout:
    return Layout(
        root=root,
        files_dir=root / "files",
        manifest_dir=root / "manifest",
        logs_dir=root / "logs",
        status_dir=root / "status",
        manifest_csv=root / "manifest" / "aia_5min_manifest.csv",
        manifest_json=root / "manifest" / "aia_5min_manifest.json",
        readme_txt=root / "README.txt",
        log_jsonl=root / "logs" / "download_log.jsonl",
        summary_json=root / "status" / "download_summary.json",
    )


def ensure_layout(layout: Layout) -> None:
    for directory in (layout.root, layout.files_dir, layout.manifest_dir, layout.logs_dir, layout.status_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        (layout.files_dir / band).mkdir(parents=True, exist_ok=True)


def get_series_for_band(band: str) -> str:
    return UV_SERIES if band == "1600" else EUV_SERIES


def build_recordset(series: str, band: str, start: datetime, end: datetime) -> str:
    duration_seconds = int((end - start).total_seconds())
    return f"{series}[{start:%Y.%m.%d_%H:%M:%S}_TAI/{duration_seconds}s][{band}]"


def iter_target_times(start: datetime, end: datetime, step_minutes: int) -> list[datetime]:
    times: list[datetime] = []
    current = start
    while current <= end:
        times.append(current)
        current += timedelta(minutes=step_minutes)
    return times


def build_filename(band: str, record_time: datetime) -> str:
    return f"aia.lev1.{band}A_{record_time:%Y_%m_%dT%H_%M_%S}.00Z.image_lev1.fits"


def query_records_for_band(client: object, band: str, start: datetime, end: datetime) -> list[dict]:
    recordset = build_recordset(get_series_for_band(band), band, start, end)
    keys, segs = client.query(recordset, key="T_REC,QUALITY", seg="image")
    rows: list[dict] = []
    for idx in range(len(keys)):
        quality = int(keys.iloc[idx]["QUALITY"])
        if quality != 0:
            continue
        record_time = parse_utc(str(keys.iloc[idx]["T_REC"]))
        segment_path = str(segs.iloc[idx]["image"])
        rows.append(
            {
                "band": band,
                "series": get_series_for_band(band),
                "record_time": record_time,
                "segment_path": segment_path,
            }
        )
    rows.sort(key=lambda item: item["record_time"])
    return rows


def pick_nearest_record(records: list[dict], target_time: datetime) -> dict:
    record_times = [item["record_time"] for item in records]
    idx = bisect_left(record_times, target_time)
    candidates: list[dict] = []
    if idx < len(records):
        candidates.append(records[idx])
    if idx > 0:
        candidates.append(records[idx - 1])
    best = min(candidates, key=lambda item: abs((item["record_time"] - target_time).total_seconds()))
    return best


def build_manifest_rows(
    start: datetime,
    end: datetime,
    step_minutes: int,
    bands: list[str],
    mirrors: list[str],
) -> list[dict]:
    import drms

    client = drms.Client()
    target_times = iter_target_times(start, end, step_minutes)
    rows: list[dict] = []
    for band in bands:
        records = query_records_for_band(client, band, start, end)
        if not records:
            continue
        for target_time in target_times:
            record = pick_nearest_record(records, target_time)
            filename = build_filename(band, record["record_time"])
            relative_path = Path("files") / band / filename
            rows.append(
                {
                    "band": band,
                    "series": record["series"],
                    "target_time_utc": format_utc(target_time),
                    "record_time_utc": format_utc(record["record_time"]),
                    "offset_seconds": int((record["record_time"] - target_time).total_seconds()),
                    "segment_path": record["segment_path"],
                    "primary_url": mirrors[0] + record["segment_path"],
                    "secondary_url": mirrors[1] + record["segment_path"],
                    "relative_path": str(relative_path).replace("\\", "/"),
                    "filename": filename,
                }
            )
    rows.sort(key=lambda item: (item["target_time_utc"], int(item["band"])))
    return rows


def write_manifest(
    layout: Layout,
    start: datetime,
    end: datetime,
    step_minutes: int | None,
    mirrors: list[str],
    rows: list[dict],
) -> None:
    ensure_layout(layout)
    fieldnames = [
        "band",
        "series",
        "target_time_utc",
        "record_time_utc",
        "offset_seconds",
        "segment_path",
        "primary_url",
        "secondary_url",
        "relative_path",
        "filename",
    ]
    with layout.manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_utc": format_utc(utc_now_naive()),
        "start_utc": format_utc(start),
        "end_utc": format_utc(end),
        "step_minutes": step_minutes,
        "bands": BANDS,
        "mirrors": mirrors,
        "row_count": len(rows),
        "expected_file_count": len(rows),
    }
    layout.manifest_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    layout.readme_txt.write_text(build_readme_text(layout, summary), encoding="utf-8-sig")


def read_manifest(layout: Layout) -> list[dict]:
    with layout.manifest_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_readme_text(layout: Layout, summary: dict) -> str:
    cadence_line = f"Cadence: every {summary['step_minutes']} minutes"
    build_line = f"1. Build manifest: python scripts/download_aia_5min_archive.py build-manifest --dataset-root \"{layout.root}\""
    return "\n".join(
        [
            "AIA 5-minute archive dataset",
            "",
            f"Root: {layout.root}",
            f"Time range: {summary['start_utc']} to {summary['end_utc']}",
            cadence_line,
            f"Bands: {', '.join(summary['bands'])}",
            f"Expected files: {summary['expected_file_count']}",
            "",
            "Directory layout:",
            f"- {layout.manifest_dir}",
            f"- {layout.files_dir}/<band>",
            f"- {layout.logs_dir}",
            f"- {layout.status_dir}",
            "",
            "Typical usage:",
            build_line,
            f"2. Download all files: python scripts/download_aia_5min_archive.py download --dataset-root \"{layout.root}\"",
            f"3. Resume later: python scripts/download_aia_5min_archive.py download --dataset-root \"{layout.root}\"",
            f"4. Check status: python scripts/download_aia_5min_archive.py status --dataset-root \"{layout.root}\"",
            "",
            "Network behavior:",
            "- Downloads use direct JSOC mirror URLs.",
            "- Direct connections are used by default; pass --use-env-proxy when a proxy is required.",
            "- If the network drops or you switch networks, rerun the same download command. Completed FITS files are skipped automatically.",
        ]
    )


def verify_fits_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with fits.open(path) as hdus:
            hdus[0].verify("fix")
            return len(hdus) >= 2 and hdus[1].data is not None
    except Exception:
        return False


def create_session(use_env_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = use_env_proxy
    session.headers.update({"User-Agent": "TIANGOU-aia-archive/1.0"})
    return session


def log_event(layout: Layout, event: dict) -> None:
    ensure_layout(layout)
    with layout.log_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def download_one_file(url: str, destination: Path, use_env_proxy: bool, timeout_connect: int, timeout_read: int) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".part")
    downloaded_bytes = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {}
    mode = "wb"
    if downloaded_bytes > 0:
        headers["Range"] = f"bytes={downloaded_bytes}-"
        mode = "ab"

    session = create_session(use_env_proxy=use_env_proxy)
    try:
        with session.get(url, stream=True, timeout=(timeout_connect, timeout_read), headers=headers) as response:
            if response.status_code == 200 and downloaded_bytes > 0:
                temp_path.unlink(missing_ok=True)
                downloaded_bytes = 0
                mode = "wb"
            response.raise_for_status()
            with temp_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    finally:
        session.close()

    temp_path.replace(destination)
    if not verify_fits_file(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded FITS failed verification: {destination}")


def download_manifest(
    layout: Layout,
    max_files: int | None,
    bands: set[str] | None,
    use_env_proxy: bool,
    timeout_connect: int,
    timeout_read: int,
    retry_wait_seconds: int,
    max_attempts_per_mirror: int,
) -> dict:
    rows = read_manifest(layout)
    pending_rows = []
    for row in rows:
        if bands is not None and row["band"] not in bands:
            continue
        destination = layout.root / row["relative_path"]
        if verify_fits_file(destination):
            continue
        pending_rows.append(row)
    if max_files is not None:
        pending_rows = pending_rows[:max_files]

    ensure_layout(layout)
    completed = 0
    failed = 0
    for index, row in enumerate(pending_rows, start=1):
        destination = layout.root / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        success = False
        urls = [row["primary_url"], row["secondary_url"]]
        for url in urls:
            for attempt in range(1, max_attempts_per_mirror + 1):
                event = {
                    "timestamp_utc": format_utc(utc_now_naive()),
                    "band": row["band"],
                    "target_time_utc": row["target_time_utc"],
                    "record_time_utc": row["record_time_utc"],
                    "relative_path": row["relative_path"],
                    "url": url,
                    "attempt": attempt,
                }
                try:
                    download_one_file(
                        url=url,
                        destination=destination,
                        use_env_proxy=use_env_proxy,
                        timeout_connect=timeout_connect,
                        timeout_read=timeout_read,
                    )
                    event["status"] = "downloaded"
                    event["bytes"] = destination.stat().st_size
                    log_event(layout, event)
                    success = True
                    break
                except Exception as exc:
                    event["status"] = "failed"
                    event["error"] = f"{type(exc).__name__}: {exc}"
                    log_event(layout, event)
                    if attempt < max_attempts_per_mirror:
                        time.sleep(retry_wait_seconds)
            if success:
                break
        if success:
            completed += 1
        else:
            failed += 1
        if index == 1 or index % 25 == 0 or index == len(pending_rows):
            print(f"[{index}/{len(pending_rows)}] completed={completed} failed={failed}")

    summary = build_status_summary(layout)
    layout.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_status_summary(layout: Layout) -> dict:
    rows = read_manifest(layout)
    total_rows = len(rows)
    complete_rows = 0
    missing_rows = 0
    total_bytes = 0
    by_band = {band: {"expected": 0, "complete": 0, "bytes": 0} for band in BANDS}
    for row in rows:
        band = row["band"]
        destination = layout.root / row["relative_path"]
        by_band[band]["expected"] += 1
        if verify_fits_file(destination):
            complete_rows += 1
            file_size = destination.stat().st_size
            total_bytes += file_size
            by_band[band]["complete"] += 1
            by_band[band]["bytes"] += file_size
        else:
            missing_rows += 1
    return {
        "dataset_root": str(layout.root),
        "manifest_csv": str(layout.manifest_csv),
        "generated_utc": format_utc(utc_now_naive()),
        "total_files_expected": total_rows,
        "files_complete": complete_rows,
        "files_missing": missing_rows,
        "bytes_downloaded": total_bytes,
        "gib_downloaded": round(total_bytes / 1024 / 1024 / 1024, 3),
        "by_band": {
            band: {
                "expected": info["expected"],
                "complete": info["complete"],
                "bytes": info["bytes"],
                "gib": round(info["bytes"] / 1024 / 1024 / 1024, 3),
            }
            for band, info in by_band.items()
        },
    }


def print_status(summary: dict) -> None:
    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Expected files: {summary['total_files_expected']}")
    print(f"Complete files: {summary['files_complete']}")
    print(f"Missing files: {summary['files_missing']}")
    print(f"Downloaded: {summary['gib_downloaded']:.3f} GiB")
    print("")
    for band in BANDS:
        info = summary["by_band"][band]
        print(f"{band:>4}  {info['complete']:>3}/{info['expected']:<3}  {info['gib']:.3f} GiB")


def main() -> None:
    parser_obj = ArgumentParser()
    subparsers = parser_obj.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    manifest_parser.add_argument("--start", default="2023-10-14T15:00:00Z")
    manifest_parser.add_argument("--end", default="2023-10-14T21:00:00Z")
    manifest_parser.add_argument("--step-minutes", type=int, default=5)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    download_parser.add_argument("--max-files", type=int, default=None)
    download_parser.add_argument("--bands", nargs="*", default=None)
    download_parser.add_argument("--use-env-proxy", action="store_true")
    download_parser.add_argument("--timeout-connect", type=int, default=30)
    download_parser.add_argument("--timeout-read", type=int, default=600)
    download_parser.add_argument("--retry-wait-seconds", type=int, default=20)
    download_parser.add_argument("--max-attempts-per-mirror", type=int, default=3)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)

    args = parser_obj.parse_args()
    layout = build_layout(args.dataset_root)

    if args.command == "build-manifest":
        start = parse_utc(args.start)
        end = parse_utc(args.end)
        rows = build_manifest_rows(
            start=start,
            end=end,
            step_minutes=args.step_minutes,
            bands=BANDS,
            mirrors=JSOC_MIRRORS,
        )
        write_manifest(
            layout,
            start=start,
            end=end,
            step_minutes=args.step_minutes,
            mirrors=JSOC_MIRRORS,
            rows=rows,
        )
        summary = build_status_summary(layout)
        layout.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Manifest written: {layout.manifest_csv}")
        print_status(summary)
        return

    if args.command == "download":
        summary = download_manifest(
            layout=layout,
            max_files=args.max_files,
            bands=set(args.bands) if args.bands else None,
            use_env_proxy=args.use_env_proxy,
            timeout_connect=args.timeout_connect,
            timeout_read=args.timeout_read,
            retry_wait_seconds=args.retry_wait_seconds,
            max_attempts_per_mirror=args.max_attempts_per_mirror,
        )
        print_status(summary)
        return

    if args.command == "status":
        summary = build_status_summary(layout)
        layout.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print_status(summary)


if __name__ == "__main__":
    main()
