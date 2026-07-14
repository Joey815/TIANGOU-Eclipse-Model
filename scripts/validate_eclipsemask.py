#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np


EXPECTED_DIMS = {"band": 23, "z": 70, "lat": 192, "lon": 288, "source": 9}
EXPECTED_SOURCES = (
    "aia_94",
    "aia_131",
    "aia_171",
    "aia_193",
    "aia_211",
    "suvi_284",
    "aia_304",
    "aia_335",
    "aia_1600",
)


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main() -> None:
    parser = ArgumentParser(description="Validate dimensions, metadata, and selected value slabs in a final 23-band mask.")
    parser.add_argument("product", type=Path)
    parser.add_argument("--expected-times", type=int)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--sample-count", type=int, default=12)
    args = parser.parse_args()

    failures: list[str] = []
    report: dict[str, object] = {"product": str(args.product.resolve())}
    if not args.product.is_file():
        raise SystemExit(f"Product does not exist: {args.product}")

    with Dataset(args.product) as ds:
        dims = {name: len(dim) for name, dim in ds.dimensions.items()}
        report["dimensions"] = dims
        for name, expected in EXPECTED_DIMS.items():
            if dims.get(name) != expected:
                failures.append(f"dimension {name}: expected {expected}, got {dims.get(name)}")
        if args.expected_times is not None and dims.get("time") != args.expected_times:
            failures.append(f"dimension time: expected {args.expected_times}, got {dims.get('time')}")

        required_variables = ("mask_euv", "source_weight", "source_id", "fism2_flux")
        missing_variables = [name for name in required_variables if name not in ds.variables]
        if missing_variables:
            failures.append("missing variables: " + ", ".join(missing_variables))

        if "source_id" in ds.variables:
            source_ids = decode_strings(np.asarray(ds.variables["source_id"][:]))
            report["source_ids"] = source_ids
            if tuple(source_ids) != EXPECTED_SOURCES:
                failures.append(f"unexpected source_id sequence: {source_ids}")

        metadata = {
            name: str(getattr(ds, name, ""))
            for name in (
                "model_name",
                "model_full_name",
                "native_resolution",
            )
        }
        report["metadata"] = metadata
        expected_metadata = {
            "model_name": "TIANGOU Eclipse Model",
            "model_full_name": "Topography-aware Irradiance And Nonuniform-Geometry Occultation Utility",
            "native_resolution": "solar image and lunar DEM pixels used without skipping",
        }
        for name, expected in expected_metadata.items():
            if metadata[name] != expected:
                failures.append(f"metadata {name}: expected {expected}, got {metadata[name]}")

        if "mask_euv" in ds.variables and all(name in dims for name in ("time", "band", "z")):
            mask = ds.variables["mask_euv"]
            sample_count = max(1, args.sample_count)
            time_indices = np.linspace(0, dims["time"] - 1, sample_count, dtype=int)
            band_indices = np.linspace(0, dims["band"] - 1, sample_count, dtype=int)
            z_indices = np.linspace(0, dims["z"] - 1, sample_count, dtype=int)
            sample_min = np.inf
            sample_max = -np.inf
            sample_nan = 0
            for time_idx, band_idx, z_idx in zip(time_indices, band_indices, z_indices):
                slab = np.asarray(mask[time_idx, band_idx, z_idx, :, :], dtype=np.float64)
                sample_nan += int(np.isnan(slab).sum())
                sample_min = min(sample_min, float(np.nanmin(slab)))
                sample_max = max(sample_max, float(np.nanmax(slab)))
            report["sample"] = {"min": sample_min, "max": sample_max, "nan_count": sample_nan}
            if sample_nan:
                failures.append(f"selected mask slabs contain {sample_nan} NaN values")
            if sample_min < -1e-6 or sample_max > 1.0 + 1e-6:
                failures.append(f"selected mask slabs outside [0,1]: min={sample_min}, max={sample_max}")

    report["ok"] = not failures
    report["failures"] = failures
    print(json.dumps(report, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
