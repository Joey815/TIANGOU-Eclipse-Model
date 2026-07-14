from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_pds3_label(label_path: str | Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw_line in Path(label_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("/*") or line.startswith("/***"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("{") or value.endswith("{"):
            continue
        if "<" in value:
            value = value.split("<", 1)[0].strip()
        value = value.strip().strip('"').strip("'")
        if value:
            meta[key] = value
    return meta


def _dtype_from_sample_type(sample_type: str, sample_bits: int) -> np.dtype:
    sample_type = sample_type.upper()
    if sample_type == "PC_REAL" and sample_bits == 32:
        return np.dtype("<f4")
    raise NotImplementedError(f"Unsupported PDS sample type: {sample_type} / {sample_bits}")


def _expected_pds_image_nbytes(meta: dict[str, str]) -> int:
    lines = int(meta["LINES"])
    samples = int(meta["LINE_SAMPLES"])
    dtype = _dtype_from_sample_type(meta["SAMPLE_TYPE"], int(meta["SAMPLE_BITS"]))
    return lines * samples * dtype.itemsize


def open_pds_float_image(img_path: str | Path, lbl_path: str | Path) -> tuple[np.memmap, dict[str, str]]:
    meta = parse_pds3_label(lbl_path)
    lines = int(meta["LINES"])
    samples = int(meta["LINE_SAMPLES"])
    dtype = _dtype_from_sample_type(meta["SAMPLE_TYPE"], int(meta["SAMPLE_BITS"]))
    img_path = Path(img_path)
    expected_nbytes = _expected_pds_image_nbytes(meta)
    actual_nbytes = img_path.stat().st_size
    if actual_nbytes != expected_nbytes:
        raise OSError(
            f"Incomplete or mismatched PDS IMG file: {img_path} "
            f"(expected {expected_nbytes} bytes, got {actual_nbytes})"
        )
    arr = np.memmap(img_path, dtype=dtype, mode="r", shape=(lines, samples))
    return arr, meta


def _iter_chunks(values: np.ndarray, block_size: int):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    for start in range(0, values.size, block_size):
        yield values[start : start + block_size]


def iter_polar_ldem_point_blocks(
    img_path: str | Path,
    lbl_path: str | Path,
    row_block_size: int = 128,
    col_block_size: int = 4096,
):
    arr, meta = open_pds_float_image(img_path=img_path, lbl_path=lbl_path)

    map_scale_km = float(meta["MAP_SCALE"]) / 1000.0
    datum_radius_km = float(meta["OFFSET"])
    center_lat_deg = float(meta["CENTER_LATITUDE"])
    min_lat_deg = float(meta["MINIMUM_LATITUDE"])
    max_lat_deg = float(meta["MAXIMUM_LATITUDE"])
    line_offset = float(meta["LINE_PROJECTION_OFFSET"])
    sample_offset = float(meta["SAMPLE_PROJECTION_OFFSET"])

    rows = np.arange(arr.shape[0], dtype=int)
    cols = np.arange(arr.shape[1], dtype=int)

    outer_lat_deg = min_lat_deg if center_lat_deg > 0 else max_lat_deg
    outer_lat_rad = np.deg2rad(abs(outer_lat_deg))
    rho_max = 2.0 * datum_radius_km * np.tan((np.pi / 2.0 - outer_lat_rad) / 2.0)

    for row_chunk in _iter_chunks(rows, row_block_size):
        row_vals = row_chunk.astype(np.float64)[:, None]
        for col_chunk in _iter_chunks(cols, col_block_size):
            col_vals = col_chunk.astype(np.float64)[None, :]

            x_km = (col_vals - sample_offset) * map_scale_km
            y_km = (line_offset - row_vals) * map_scale_km
            rho = np.hypot(x_km, y_km)
            inside = rho <= (rho_max + 1e-6)
            if not np.any(inside):
                continue

            c = 2.0 * np.arctan2(rho, 2.0 * datum_radius_km)
            if center_lat_deg > 0:
                lat_rad = np.pi / 2.0 - c
                lon_rad = np.mod(np.arctan2(x_km, -y_km), 2.0 * np.pi)
            else:
                lat_rad = -np.pi / 2.0 + c
                lon_rad = np.mod(np.arctan2(x_km, y_km), 2.0 * np.pi)

            elev_km = np.asarray(arr[np.ix_(row_chunk, col_chunk)], dtype=np.float64)
            radius = datum_radius_km + elev_km

            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)
            yield np.column_stack([x[inside], y[inside], z[inside]])


def iter_sldem_point_blocks(
    img_path: str | Path,
    lbl_path: str | Path,
    lon_bounds_deg: tuple[float, float] | None = None,
    row_block_size: int = 128,
    col_block_size: int = 4096,
):
    arr, meta = open_pds_float_image(img_path=img_path, lbl_path=lbl_path)

    map_resolution = float(meta["MAP_RESOLUTION"])
    min_lat = float(meta["MINIMUM_LATITUDE"])
    max_lat = float(meta["MAXIMUM_LATITUDE"])
    west_lon = float(meta["WESTERNMOST_LONGITUDE"])
    datum_radius_km = float(meta["OFFSET"])

    rows = np.arange(arr.shape[0], dtype=int)
    cols = np.arange(arr.shape[1], dtype=int)

    lon_deg = west_lon + (cols + 0.5) / map_resolution
    if lon_bounds_deg is not None:
        lo, hi = lon_bounds_deg
        lo = lo % 360.0
        hi = hi % 360.0
        if lo <= hi:
            keep = (lon_deg >= lo) & (lon_deg <= hi)
        else:
            keep = (lon_deg >= lo) | (lon_deg <= hi)
        cols = cols[keep]
        lon_deg = lon_deg[keep]

    lat_deg = max_lat - (rows + 0.5) / map_resolution
    lat_keep = (lat_deg >= min_lat) & (lat_deg <= max_lat)
    rows = rows[lat_keep]
    lat_deg = lat_deg[lat_keep]

    for row_chunk, lat_chunk_deg in zip(_iter_chunks(rows, row_block_size), _iter_chunks(lat_deg, row_block_size)):
        lat_rad = np.deg2rad(lat_chunk_deg).astype(np.float64)[:, None]
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        for col_chunk, lon_chunk_deg in zip(_iter_chunks(cols, col_block_size), _iter_chunks(lon_deg, col_block_size)):
            lon_rad = np.deg2rad(lon_chunk_deg).astype(np.float64)[None, :]
            cos_lon = np.cos(lon_rad)
            sin_lon = np.sin(lon_rad)

            elev_km = np.asarray(arr[np.ix_(row_chunk, col_chunk)], dtype=np.float64)
            radius = datum_radius_km + elev_km

            x = radius * cos_lat * cos_lon
            y = radius * cos_lat * sin_lon
            z = radius * sin_lat
            yield np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def iter_combined_lunar_point_blocks(
    global_img_path: str | Path,
    global_lbl_path: str | Path,
    polar_datasets: list[tuple[str | Path, str | Path]],
    lon_bounds_deg: tuple[float, float] | None = None,
    global_row_block_size: int = 128,
    global_col_block_size: int = 4096,
    polar_row_block_size: int = 128,
    polar_col_block_size: int = 4096,
):
    yield from iter_sldem_point_blocks(
        img_path=global_img_path,
        lbl_path=global_lbl_path,
        lon_bounds_deg=lon_bounds_deg,
        row_block_size=global_row_block_size,
        col_block_size=global_col_block_size,
    )
    for img_path, lbl_path in polar_datasets:
        yield from iter_polar_ldem_point_blocks(
            img_path=img_path,
            lbl_path=lbl_path,
            row_block_size=polar_row_block_size,
            col_block_size=polar_col_block_size,
        )


def discover_polar_ldem_pairs(
    polar_dir: str | Path,
) -> list[tuple[Path, Path]]:
    polar_root = Path(polar_dir).expanduser().resolve()
    pairs: list[tuple[Path, Path]] = []
    for stem in ("ldem_60n_240m_float", "ldem_60s_240m_float"):
        img_path = polar_root / f"{stem}.img"
        lbl_path = polar_root / f"{stem}.lbl"
        if not img_path.is_file() or not lbl_path.is_file():
            raise FileNotFoundError(f"Required polar LDEM pair is missing: {img_path}, {lbl_path}")
        meta = parse_pds3_label(lbl_path)
        expected_nbytes = _expected_pds_image_nbytes(meta)
        if img_path.stat().st_size != expected_nbytes:
            raise OSError(
                f"Incomplete polar LDEM file: {img_path} "
                f"(expected {expected_nbytes} bytes, got {img_path.stat().st_size})"
            )
        pairs.append((img_path, lbl_path))

    return pairs
