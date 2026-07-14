from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

KERNEL_URLS: dict[str, list[str]] = {
    "naif0012.tls": [
        f"{NAIF_BASE}/lsk/naif0012.tls",
    ],
    "pck00011.tpc": [
        f"{NAIF_BASE}/pck/pck00011.tpc",
    ],
    "de440.bsp": [
        f"{NAIF_BASE}/spk/planets/de440.bsp",
    ],
    "earth_latest_high_prec.bpc": [
        f"{NAIF_BASE}/pck/earth_latest_high_prec.bpc",
    ],
    "moon_pa_de440_200625.bpc": [
        f"{NAIF_BASE}/pck/moon_pa_de440_200625.bpc",
    ],
    "moon_de440_200625.tf": [
        f"{NAIF_BASE}/fk/satellites/moon_de440_200625.tf",
        f"{NAIF_BASE}/fk/satellites/a_old_versions/moon_de440_200625.tf",
    ],
}

TIANGOU_KERNEL_SET = (
    "naif0012.tls",
    "pck00011.tpc",
    "de440.bsp",
    "earth_latest_high_prec.bpc",
    "moon_pa_de440_200625.bpc",
    "moon_de440_200625.tf",
)


@dataclass(frozen=True)
class DownloadResult:
    name: str
    path: Path
    downloaded: bool
    size_bytes: int
    source_url: str


def _download_one(url: str, destination: Path, timeout_s: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(destination)


def download_kernel(name: str, target_dir: Path, overwrite: bool = False) -> DownloadResult:
    if name not in KERNEL_URLS:
        raise KeyError(f"Unknown kernel: {name}")

    target_dir = Path(target_dir).expanduser().resolve()
    path = target_dir / name
    if path.exists() and not overwrite and path.stat().st_size > 0:
        return DownloadResult(
            name=name,
            path=path,
            downloaded=False,
            size_bytes=path.stat().st_size,
            source_url="cached",
        )

    last_error: Exception | None = None
    for url in KERNEL_URLS[name]:
        try:
            _download_one(url, path)
            return DownloadResult(
                name=name,
                path=path,
                downloaded=True,
                size_bytes=path.stat().st_size,
                source_url=url,
            )
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc

    raise RuntimeError(f"Failed to download {name}: {last_error}") from last_error


def ensure_kernel_set(
    target_dir: str | Path,
    kernel_names: Iterable[str] = TIANGOU_KERNEL_SET,
    overwrite: bool = False,
) -> list[DownloadResult]:
    target_dir = Path(target_dir).expanduser().resolve()
    results: list[DownloadResult] = []
    for name in kernel_names:
        results.append(download_kernel(name=name, target_dir=target_dir, overwrite=overwrite))
    return results
