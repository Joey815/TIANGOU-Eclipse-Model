#!/usr/bin/env python3

"""Build a FISM2-weighted 23-band eclipse mask from AIA/SUVI source masks.

This is a streamed post-processing product:

    FISM2 high-resolution spectrum + AIA/SUVI channel masks
    -> mask_euv(time, band, z, lat, lon)

The source channel cubes store ``source_mask(time, height, lat, lon)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


HC = 6.62607015e-34 * 299792458.0
MODEL_NAME = "TIANGOU Eclipse Model"
MODEL_FULL_NAME = "Topography-aware Irradiance And Nonuniform-Geometry Occultation Utility"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_MASK_ROOT = REPO_ROOT / "data" / "output" / "source_masks"
DEFAULT_FISM_ROOT = REPO_ROOT / "data" / "fism2" / "by_date"
DEFAULT_Z_FILE = DEFAULT_PIPELINE_ROOT / "vertical_grids" / "waccm6_70_heights_km.txt"
DEFAULT_CROSS_SECTION_FILE = DEFAULT_PIPELINE_ROOT / "stan_bands" / "cross_sections.dat"


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    instrument: str
    wavelength_angstrom: str

    @property
    def directory_token(self) -> str:
        if self.instrument == "aia":
            return self.wavelength_angstrom
        if self.instrument == "suvi":
            return f"suvi_ci{int(self.wavelength_angstrom):03d}"
        raise ValueError(f"Unsupported instrument: {self.instrument}")


SOURCE_CONFIGS = [
    SourceConfig("aia_94", "aia", "94"),
    SourceConfig("aia_131", "aia", "131"),
    SourceConfig("aia_171", "aia", "171"),
    SourceConfig("aia_193", "aia", "193"),
    SourceConfig("aia_211", "aia", "211"),
    SourceConfig("suvi_284", "suvi", "284"),
    SourceConfig("aia_304", "aia", "304"),
    SourceConfig("aia_335", "aia", "335"),
]

AIA1600_SOURCE = SourceConfig("aia_1600", "aia", "1600")

ANCHORS_NM = np.array([9.4, 13.1, 17.1, 19.3, 21.1, 28.4, 30.4, 33.5], dtype=np.float64)
ANCHOR_SOURCES = ["aia_94", "aia_131", "aia_171", "aia_193", "aia_211", "suvi_284", "aia_304", "aia_335"]

STAN_BAND_CLASSES = [
    "interval",
    "interval",
    "interval",
    "interval",
    "interval",
    "interval",
    "interval",
    "interval",
    "interval",
    "interval",
    "interval",
    "n2_lt31_low_medium",
    "n2_ge31_high",
    "n2_lt4_low",
    "n2_ge4_lt31_medium",
    "n2_ge31_high",
    "n2_lt4_low",
    "n2_ge4_lt31_medium",
    "n2_ge31_high",
    "interval",
    "interval",
    "interval",
    "interval_fism2_105_121",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a FISM2-weighted multichannel 23-band WACCM eclipse mask."
    )
    parser.add_argument("--event", default="20240408", help="Event date as YYYYMMDD")
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--fism-root", type=Path, default=DEFAULT_FISM_ROOT)
    parser.add_argument("--z-file", type=Path, default=DEFAULT_Z_FILE)
    parser.add_argument("--fism-hr", type=Path, default=None, help="Override FISM2 flare_hr NetCDF")
    parser.add_argument("--fism-bands", type=Path, default=None, help="Override FISM2 flare_bands NetCDF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-sec", type=int, default=15 * 3600)
    parser.add_argument("--end-sec", type=int, default=21 * 3600)
    parser.add_argument("--step-sec", type=int, default=300)
    parser.add_argument(
        "--source-start-sec",
        type=int,
        default=15 * 3600,
        help="First time index in the source mask files, in seconds after midnight UT",
    )
    parser.add_argument("--cross-section-file", type=Path, default=DEFAULT_CROSS_SECTION_FILE)
    parser.add_argument("--compression-level", type=int, default=4)
    return parser.parse_args()


def event_to_date(event: str) -> datetime:
    return datetime.strptime(event, "%Y%m%d")


def event_label(event: str) -> str:
    dt = event_to_date(event)
    return f"{dt.year:04d}_{dt.month:02d}_{dt.day:02d}"


def event_iso(event: str) -> str:
    dt = event_to_date(event)
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"


def find_fism_file(fism_root: Path, event: str, product: str, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(override)
        return override

    subdir = "flare_hr" if product == "hr" else "flare_bands"
    pattern = "FISM_60sec_*.nc" if product == "hr" else "FISM_bands_*.nc"
    base = fism_root / event_iso(event) / subdir / "netcdf"
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {product} FISM2 file found under {base}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one {product} FISM2 file under {base}, found {len(matches)}")
    return matches[0]


def find_source_file(mask_root: Path, event: str, source: SourceConfig) -> Path:
    run_dir = mask_root / f"{event}_{source.directory_token}_5min_waccm6_70_tiangou"
    path = run_dir / "TIANGOU_SourceMask.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_datesec(start_sec: int, end_sec: int, step_sec: int) -> np.ndarray:
    return np.arange(start_sec, end_sec + 1, step_sec, dtype=np.int32)


def load_fism_hr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path, "r") as ds:
        wavelength = np.asarray(ds.variables["wavelength"][:], dtype=np.float64)
        utc = np.asarray(ds.variables["utc"][:], dtype=np.int32)
        irradiance = np.asarray(ds.variables["irradiance"][:], dtype=np.float64)
    return wavelength, utc, irradiance


def load_fism_bands(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path, "r") as ds:
        center = np.asarray(ds.variables["wavelength"][:], dtype=np.float64)
        width = np.asarray(ds.variables["band_width"][:], dtype=np.float64)
        seconds = np.asarray(ds.variables["date_sec"][:], dtype=np.int32)
        ssi = np.asarray(ds.variables["ssi"][:], dtype=np.float64)
    return center, width, seconds, ssi


def load_z_values(path: Path) -> np.ndarray:
    # The WACCM6_70 file is stored top-to-bottom. Existing lonfix 23-band
    # products expose z bottom-to-top, so keep that convention here.
    z = np.sort(np.loadtxt(path, dtype=np.float32))
    return z


def build_source_time_indices(datesec: np.ndarray, source_start_sec: int, step_sec: int) -> np.ndarray:
    offsets = datesec.astype(np.int64) - int(source_start_sec)
    if np.any(offsets < 0) or np.any(offsets % int(step_sec) != 0):
        raise ValueError(
            "Requested datesec values are not aligned with source mask time axis: "
            f"source_start_sec={source_start_sec}, step_sec={step_sec}, datesec={datesec[:5]}"
        )
    return (offsets // int(step_sec)).astype(np.int64)


def build_lat_lon(src_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, nlat, nlon = src_shape
    lat = np.arange(-90.0, -90.0 + 0.9375 * nlat, 0.9375, dtype=np.float32)
    lon_native = np.arange(-180.0, -180.0 + 1.25 * nlon, 1.25, dtype=np.float32)
    lon_east = np.mod(lon_native, 360.0).astype(np.float32)
    lon_order = np.argsort(lon_east)
    lon = lon_east[lon_order]
    return lat, lon, lon_order


def build_source_weights(
    wavelength_nm: np.ndarray,
    source_ids: list[str],
) -> np.ndarray:
    source_index = {name: idx for idx, name in enumerate(source_ids)}
    weights = np.zeros((wavelength_nm.size, len(source_ids)), dtype=np.float64)
    log_anchors = np.log(ANCHORS_NM)

    def add(row: int, source_id: str, value: float) -> None:
        weights[row, source_index[source_id]] += value

    for row, lam in enumerate(wavelength_nm):
        if lam < ANCHORS_NM[0]:
            add(row, "aia_94", 0.5)
            add(row, "aia_131", 0.5)
            continue

        if lam > ANCHORS_NM[-1]:
            add(row, "aia_1600", 1.0)
            continue

        upper = int(np.searchsorted(ANCHORS_NM, lam, side="right"))
        lower = max(0, upper - 1)
        upper = min(upper, len(ANCHORS_NM) - 1)
        if lower == upper or np.isclose(lam, ANCHORS_NM[lower]):
            add(row, ANCHOR_SOURCES[lower], 1.0)
            continue

        x0 = log_anchors[lower]
        x1 = log_anchors[upper]
        frac = float((np.log(lam) - x0) / (x1 - x0))
        add(row, ANCHOR_SOURCES[lower], 1.0 - frac)
        add(row, ANCHOR_SOURCES[upper], frac)

    row_sums = weights.sum(axis=1)
    bad = np.where(~np.isfinite(row_sums) | (row_sums <= 0.0))[0]
    if bad.size:
        raise ValueError(f"Invalid source weights at wavelength indices: {bad[:10]}")
    weights /= row_sums[:, np.newaxis]
    return weights


def load_n2_cross_section_megabarn(path: Path, wavelength_nm: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    table = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 4:
        raise ValueError(f"Unexpected cross section table shape in {path}: {table.shape}")
    table_wavelength = table[:, 0]
    n2_megabarn = table[:, 3]
    return np.interp(wavelength_nm, table_wavelength, n2_megabarn).astype(np.float64)


def build_stan_membership(
    wavelength_nm: np.ndarray,
    band_center: np.ndarray,
    band_width: np.ndarray,
    n2_sigma_megabarn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    membership = np.zeros((band_center.size, wavelength_nm.size), dtype=bool)
    classes = np.asarray(STAN_BAND_CLASSES[: band_center.size], dtype=object)

    if band_center.size != len(STAN_BAND_CLASSES):
        raise ValueError(f"Expected 23 FISM2 Stan bands, found {band_center.size}")

    for band_idx, (center, width) in enumerate(zip(band_center, band_width)):
        low = center - 0.5 * width
        high = center + 0.5 * width
        interval = (wavelength_nm >= low) & (wavelength_nm < high)
        if band_idx == band_center.size - 1:
            interval = (wavelength_nm >= low) & (wavelength_nm <= high)

        if band_idx == 11:
            membership[band_idx, :] = interval & (n2_sigma_megabarn < 31.0)
        elif band_idx == 12:
            membership[band_idx, :] = interval & (n2_sigma_megabarn >= 31.0)
        elif band_idx in (13, 16):
            membership[band_idx, :] = interval & (n2_sigma_megabarn < 4.0)
        elif band_idx in (14, 17):
            membership[band_idx, :] = interval & (n2_sigma_megabarn >= 4.0) & (n2_sigma_megabarn < 31.0)
        elif band_idx in (15, 18):
            membership[band_idx, :] = interval & (n2_sigma_megabarn >= 31.0)
        else:
            membership[band_idx, :] = interval

        if not np.any(membership[band_idx, :]):
            raise ValueError(
                f"No FISM2 wavelength samples assigned to Stan band {band_idx + 1} "
                f"({classes[band_idx]})"
            )

    return membership, classes


def photon_weights(wavelength_nm: np.ndarray) -> np.ndarray:
    delta_nm = np.empty_like(wavelength_nm, dtype=np.float64)
    delta_nm[:-1] = np.diff(wavelength_nm)
    delta_nm[-1] = delta_nm[-2]
    return delta_nm * (wavelength_nm * 1.0e-9) / HC / 1.0e4


def compute_band_source_weights(
    datesec: np.ndarray,
    fism_wavelength: np.ndarray,
    fism_utc: np.ndarray,
    fism_irradiance: np.ndarray,
    band_center: np.ndarray,
    band_width: np.ndarray,
    source_lambda_weights: np.ndarray,
    stan_membership: np.ndarray,
) -> np.ndarray:
    source_count = source_lambda_weights.shape[1]
    out = np.zeros((datesec.size, band_center.size, source_count), dtype=np.float32)
    q = photon_weights(fism_wavelength)

    utc_to_index = {int(sec): idx for idx, sec in enumerate(fism_utc)}
    for time_idx, sec in enumerate(datesec):
        if int(sec) not in utc_to_index:
            nearest = int(np.argmin(np.abs(fism_utc - sec)))
            spec = fism_irradiance[nearest, :]
        else:
            spec = fism_irradiance[utc_to_index[int(sec)], :]

        for band_idx, (center, width) in enumerate(zip(band_center, band_width)):
            mask = stan_membership[band_idx, :]
            if not np.any(mask):
                low = center - 0.5 * width
                high = center + 0.5 * width
                raise ValueError(f"No FISM2 wavelength samples in band {band_idx + 1}: {low}-{high} nm")

            spectral_weight = spec[mask] * q[mask]
            denom = float(np.nansum(spectral_weight))
            if denom <= 0.0 or not np.isfinite(denom):
                out[time_idx, band_idx, :] = np.float32(1.0 / source_count)
                continue
            numer = spectral_weight @ source_lambda_weights[mask, :]
            out[time_idx, band_idx, :] = (numer / denom).astype(np.float32)

    return out


def select_official_flux(
    datesec: np.ndarray,
    fism_band_seconds: np.ndarray,
    fism_band_ssi: np.ndarray,
) -> np.ndarray:
    sec_to_index = {int(sec): idx for idx, sec in enumerate(fism_band_seconds)}
    rows = []
    for sec in datesec:
        if int(sec) in sec_to_index:
            rows.append(fism_band_ssi[sec_to_index[int(sec)], :])
        else:
            nearest = int(np.argmin(np.abs(fism_band_seconds - sec)))
            rows.append(fism_band_ssi[nearest, :])
    return np.asarray(rows, dtype=np.float32)


def create_time_values(event: str, datesec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    base = event_to_date(event)
    time_units = f"days since {base:%Y-%m-%d %H:%M:%S}"
    time_values = datesec.astype(np.float64) / 86400.0
    date_values = np.full(datesec.size, int(event), dtype=np.int32)
    return time_values, date_values, datesec.astype(np.int32), time_units


def main() -> None:
    args = parse_args()
    z_values = load_z_values(args.z_file)
    datesec = build_datesec(args.start_sec, args.end_sec, args.step_sec)
    source_configs = list(SOURCE_CONFIGS) + [AIA1600_SOURCE]
    source_ids = [source.source_id for source in source_configs]
    source_files = [find_source_file(args.mask_root, args.event, source) for source in source_configs]
    fism_hr = find_fism_file(args.fism_root, args.event, "hr", args.fism_hr)
    fism_bands = find_fism_file(args.fism_root, args.event, "bands", args.fism_bands)

    fism_wavelength, fism_utc, fism_irradiance = load_fism_hr(fism_hr)
    band_center, band_width, band_seconds, band_ssi = load_fism_bands(fism_bands)
    n2_sigma_megabarn = load_n2_cross_section_megabarn(args.cross_section_file, fism_wavelength)
    stan_membership, band_classes = build_stan_membership(
        fism_wavelength,
        band_center,
        band_width,
        n2_sigma_megabarn,
    )
    source_lambda_weights = build_source_weights(fism_wavelength, source_ids)
    source_weight = compute_band_source_weights(
        datesec,
        fism_wavelength,
        fism_utc,
        fism_irradiance,
        band_center,
        band_width,
        source_lambda_weights,
        stan_membership,
    )
    fism2_flux = select_official_flux(datesec, band_seconds, band_ssi)
    source_time_indices = build_source_time_indices(datesec, args.source_start_sec, args.step_sec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_datasets = [Dataset(path, "r") for path in source_files]
    try:
        source_vars = [ds.variables["source_mask"] for ds in source_datasets]
        source_shape = source_vars[0].shape
        if len(source_shape) != 4:
            raise ValueError(f"Expected source mask to be 4D, got {source_shape}")
        for path, var in zip(source_files, source_vars):
            if var.shape != source_shape:
                raise ValueError(f"Source shape mismatch for {path}: {var.shape} vs {source_shape}")
        if np.max(source_time_indices) >= source_shape[0]:
            raise ValueError(
                f"Source masks have {source_shape[0]} times, but requested source index "
                f"{int(np.max(source_time_indices))}"
            )
        if source_shape[1] < z_values.size:
            raise ValueError(f"Source masks have {source_shape[1]} heights, requested {z_values.size}")

        lat, lon, lon_order = build_lat_lon(source_shape)
        time_values, date_values, datesec_values, time_units = create_time_values(args.event, datesec)
        source_height_desc = source_shape[1] - 1
        input_height_indices = source_height_desc - np.arange(z_values.size, dtype=np.int64)

        with Dataset(args.output, "w", format="NETCDF4") as dst:
            dst.createDimension("time", None)
            dst.createDimension("band", band_center.size)
            dst.createDimension("z", z_values.size)
            dst.createDimension("lat", lat.size)
            dst.createDimension("lon", lon.size)
            dst.createDimension("source", len(source_ids))
            dst.createDimension("wavelength_hr", fism_wavelength.size)

            time_var = dst.createVariable("time", "f8", ("time",))
            time_var.units = time_units
            time_var.calendar = "noleap"
            time_var.long_name = "time"
            time_var[:] = time_values

            date_var = dst.createVariable("date", "i4", ("time",))
            date_var.units = "YYYYMMDD"
            date_var[:] = date_values

            datesec_var = dst.createVariable("datesec", "i4", ("time",))
            datesec_var.units = "seconds after midnight UT"
            datesec_var[:] = datesec_values

            band_var = dst.createVariable("band", "i4", ("band",))
            band_var.long_name = "FISM2 Stan band index"
            band_var[:] = np.arange(1, band_center.size + 1, dtype=np.int32)

            wavelength_var = dst.createVariable("wavelength", "f4", ("band",))
            wavelength_var.units = "nm"
            wavelength_var.long_name = "FISM2 Stan band center wavelength"
            wavelength_var[:] = band_center.astype(np.float32)

            band_width_var = dst.createVariable("band_width", "f4", ("band",))
            band_width_var.units = "nm"
            band_width_var.long_name = "FISM2 Stan band width"
            band_width_var[:] = band_width.astype(np.float32)

            low_var = dst.createVariable("band_low", "f4", ("band",))
            low_var.units = "nm"
            low_var[:] = (band_center - 0.5 * band_width).astype(np.float32)

            high_var = dst.createVariable("band_high", "f4", ("band",))
            high_var.units = "nm"
            high_var[:] = (band_center + 0.5 * band_width).astype(np.float32)

            class_var = dst.createVariable("band_class", str, ("band",))
            class_var.long_name = "Stan band interval or official N2 absorption class"
            class_var[:] = band_classes

            membership_count_var = dst.createVariable("stan_membership_count", "i4", ("band",))
            membership_count_var.long_name = "Number of FISM2 0.1 nm wavelength samples assigned to each band"
            membership_count_var[:] = stan_membership.sum(axis=1).astype(np.int32)

            fism_wave_var = dst.createVariable("fism_wavelength", "f4", ("wavelength_hr",))
            fism_wave_var.units = "nm"
            fism_wave_var.long_name = "FISM2 high-rate wavelength centers used for source weighting"
            fism_wave_var[:] = fism_wavelength.astype(np.float32)

            n2_var = dst.createVariable("n2_sigma_megabarn", "f4", ("wavelength_hr",))
            n2_var.units = "megabarn"
            n2_var.long_name = "N2 absorption cross section interpolated to FISM2 wavelengths"
            n2_var[:] = n2_sigma_megabarn.astype(np.float32)

            membership_var = dst.createVariable("stan_membership", "i1", ("band", "wavelength_hr"))
            membership_var.units = "1"
            membership_var.long_name = "FISM2 wavelength membership for each Stan band after repeated-band classification"
            membership_var[:] = stan_membership.astype(np.int8)

            z_var = dst.createVariable("z", "f4", ("z",))
            z_var.units = "km"
            z_var.positive = "up"
            z_var[:] = z_values

            lat_var = dst.createVariable("lat", "f4", ("lat",))
            lat_var.units = "degrees_north"
            lat_var[:] = lat

            lon_var = dst.createVariable("lon", "f4", ("lon",))
            lon_var.units = "degrees_east"
            lon_var[:] = lon

            source_var = dst.createVariable("source_id", str, ("source",))
            source_var[:] = np.asarray(source_ids, dtype=object)

            weight_var = dst.createVariable("source_weight", "f4", ("time", "band", "source"))
            weight_var.units = "1"
            weight_var.long_name = "FISM2 spectral-weighted contribution of each source mask"
            weight_var[:] = source_weight

            flux_var = dst.createVariable("fism2_flux", "f4", ("time", "band"))
            flux_var.units = "photons cm-2 s-1"
            flux_var.long_name = "Official FISM2 flare Stan-band flux for the selected times"
            flux_var[:] = fism2_flux

            mask_var = dst.createVariable(
                "mask_euv",
                "f4",
                ("time", "band", "z", "lat", "lon"),
                zlib=True,
                complevel=args.compression_level,
                shuffle=True,
                chunksizes=(1, band_center.size, 1, lat.size, lon.size),
            )
            mask_var.units = "1"
            mask_var.valid_min = np.float32(0.0)
            mask_var.valid_max = np.float32(1.0)
            mask_var.long_name = "FISM2-weighted multichannel EUV eclipse attenuation factor"

            dst.title = f"{MODEL_NAME}: FISM2-weighted multichannel eclipse mask for WACCM EUV bands"
            dst.model_name = MODEL_NAME
            dst.model_full_name = MODEL_FULL_NAME
            dst.native_resolution = "solar image and lunar DEM pixels used without skipping"
            dst.event = args.event
            dst.fism2_hr_file = str(fism_hr)
            dst.fism2_bands_file = str(fism_bands)
            dst.source_mask_files = "\n".join(str(path) for path in source_files)
            dst.cross_section_file = str(args.cross_section_file)
            dst.history = (
                f"{datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC: created by "
                "make_tiangou_23band_mask.py"
            )

            slabs = np.empty((len(source_ids), lat.size, lon.size), dtype=np.float32)
            source_time_slabs = np.empty((len(source_configs), z_values.size, lat.size, lon.size), dtype=np.float32)
            for time_idx in range(datesec.size):
                print(f"[{time_idx + 1}/{datesec.size}] time seconds={int(datesec[time_idx])}")
                source_time_idx = int(source_time_indices[time_idx])
                for source_idx, var in enumerate(source_vars):
                    source_cube = np.asarray(var[source_time_idx, :, :, :], dtype=np.float32)
                    source_cube = source_cube[input_height_indices, :, :]
                    if not np.all(np.isfinite(source_cube)):
                        raise ValueError(
                            f"Non-finite values in source mask {source_files[source_idx]} "
                            f"at source time index {source_time_idx}"
                        )
                    source_min = float(np.min(source_cube))
                    source_max = float(np.max(source_cube))
                    if source_min < -1e-6 or source_max > 1.0 + 1e-6:
                        raise ValueError(
                            f"Source mask outside [0, 1] in {source_files[source_idx]}: "
                            f"min={source_min}, max={source_max}"
                        )
                    np.clip(source_cube, 0.0, 1.0, out=source_cube)
                    source_time_slabs[source_idx, :, :, :] = source_cube[:, :, lon_order]

                for z_idx in range(z_values.size):
                    slabs[:, :, :] = source_time_slabs[:, z_idx, :, :]
                    band_slabs = np.tensordot(source_weight[time_idx, :, :], slabs, axes=(1, 0))
                    np.clip(band_slabs, 0.0, 1.0, out=band_slabs)
                    mask_var[time_idx, :, z_idx, :, :] = band_slabs.astype(np.float32)
    finally:
        for ds in source_datasets:
            ds.close()

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
