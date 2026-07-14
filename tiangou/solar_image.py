from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from astropy.io import fits
from dateutil import parser as date_parser
import numpy as np


PIXELS_PER_RADIAN_ARCSEC = 180.0 * 3600.0 / np.pi
_AIA_NAME = re.compile(
    r"(?P<wl>\d{2,4})A_(?P<stamp>\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})",
    re.IGNORECASE,
)
_SUVI_NAME = re.compile(
    r"(?:ci)?(?P<wl>\d{3}).*?_g(?P<satellite>\d{2})_s(?P<stamp>\d{8}T\d{6})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SolarImage:
    data: np.ndarray
    center_column: float
    center_row: float
    pixels_per_radian: float
    observation_time: datetime
    source_path: Path


def _parse_filename(path: Path, instrument: str) -> tuple[int, int | None, datetime] | None:
    if instrument == "aia":
        match = _AIA_NAME.search(path.name)
        if match is None:
            return None
        return (
            int(match.group("wl")),
            None,
            datetime.strptime(match.group("stamp"), "%Y_%m_%dT%H_%M_%S"),
        )

    match = _SUVI_NAME.search(path.name)
    if match is None:
        return None
    return (
        int(match.group("wl")),
        int(match.group("satellite")),
        datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%S"),
    )


def _fits_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.fits", "*.fit", "*.fts"):
        files.extend(folder.glob(pattern))
    return sorted(set(files))


def select_solar_file(
    folder: str | Path,
    instrument: str,
    wavelength: int,
    target_time: datetime,
    satellite_number: int = 16,
) -> Path:
    instrument = instrument.lower()
    if instrument not in {"aia", "suvi"}:
        raise ValueError(f"Unsupported instrument: {instrument}")

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    candidates: list[tuple[datetime, Path]] = []
    for path in _fits_files(root):
        parsed = _parse_filename(path, instrument)
        if parsed is None:
            continue
        file_wavelength, file_satellite, file_time = parsed
        if file_wavelength != int(wavelength):
            continue
        if instrument == "suvi" and file_satellite != int(satellite_number):
            continue
        candidates.append((file_time, path))

    if not candidates:
        raise FileNotFoundError(
            f"No {instrument.upper()} {wavelength} FITS files found in {root}"
        )

    file_time, path = min(candidates, key=lambda item: abs(item[0] - target_time))
    if abs((file_time - target_time).total_seconds()) > 12 * 3600:
        raise FileNotFoundError(
            f"No {instrument.upper()} {wavelength} image within 12 hours of {target_time}"
        )
    return path


def _image_hdu(hdus: fits.HDUList):
    for hdu in hdus:
        if hdu.data is None:
            continue
        array = np.asarray(hdu.data).squeeze()
        if array.ndim == 2:
            return hdu, array
    raise ValueError("FITS file contains no two-dimensional image")


def load_solar_image(
    folder: str | Path,
    instrument: str,
    wavelength: int,
    target_time: datetime,
    satellite_number: int = 16,
) -> SolarImage:
    source_path = select_solar_file(
        folder=folder,
        instrument=instrument,
        wavelength=wavelength,
        target_time=target_time,
        satellite_number=satellite_number,
    )

    with fits.open(source_path, memmap=True) as hdus:
        hdu, raw = _image_hdu(hdus)
        header = hdus[0].header.copy()
        header.extend(hdu.header, update=True)
        image = np.asarray(raw, dtype=np.float32).copy()

    image[~np.isfinite(image)] = 0.0
    image[image < 0.0] = 0.0
    image[image > 65536.0] = 0.0

    date_value = header.get("DATE-OBS", header.get("DATE_OBS", header.get("DATE")))
    if date_value is None:
        raise KeyError(f"Missing observation time in {source_path}")
    observation_time = date_parser.parse(str(date_value))
    if observation_time.tzinfo is not None:
        observation_time = observation_time.astimezone(timezone.utc).replace(tzinfo=None)

    pixel_arcsec = abs(float(header["CDELT1"]))
    if pixel_arcsec <= 0.0:
        raise ValueError(f"Invalid CDELT1 in {source_path}: {pixel_arcsec}")

    return SolarImage(
        data=np.ascontiguousarray(image),
        center_column=float(header["CRPIX1"]),
        center_row=float(header["CRPIX2"]),
        pixels_per_radian=PIXELS_PER_RADIAN_ARCSEC / pixel_arcsec,
        observation_time=observation_time,
        source_path=source_path,
    )
