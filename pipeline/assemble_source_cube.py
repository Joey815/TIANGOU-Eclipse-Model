import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def extract_height_from_filename(filename: str) -> float:
    return float(filename.split("_")[1].replace("km", ""))


def extract_time_from_filename(filename: str) -> datetime:
    timestamp_str = filename.split("_")[0][:12]
    return datetime.strptime(timestamp_str, "%Y%m%d%H%M")


def build_stack(input_dir: Path) -> np.ndarray:
    file_list = sorted(p.name for p in input_dir.glob("*.nc"))
    if not file_list:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}")

    height_list = sorted({extract_height_from_filename(name) for name in file_list}, reverse=True)
    file_times = sorted({extract_time_from_filename(name) for name in file_list})

    all_data = []
    n_lat = None
    n_lon = None

    for current_time in file_times:
        time_step_data = []

        for height in height_list:
            matches = [
                name for name in file_list
                if abs(extract_height_from_filename(name) - height) < 1e-6
                and extract_time_from_filename(name) == current_time
            ]

            if not matches:
                raise FileNotFoundError(
                    f"Missing snapshot for time={current_time:%Y-%m-%dT%H:%M:%S}, "
                    f"height={height:.2f} km"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple snapshots for time={current_time:%Y-%m-%dT%H:%M:%S}, "
                    f"height={height:.2f} km: {matches}"
                )

            file_path = input_dir / matches[0]
            with Dataset(file_path, "r") as nc_file:
                data = nc_file.variables["transmission"][:]
                data = np.asarray(data)

            if n_lat is None or n_lon is None:
                if data.shape == (192, 288):
                    n_lat, n_lon = data.shape
                elif data.shape == (288, 192):
                    data = data.T
                    n_lat, n_lon = data.shape
                else:
                    n_lat, n_lon = data.shape
            elif data.shape != (n_lat, n_lon):
                if data.T.shape == (n_lat, n_lon):
                    data = data.T
                else:
                    raise ValueError(
                        f"Inconsistent array shape in {file_path}: {data.shape}, "
                        f"expected {(n_lat, n_lon)}"
                    )

            time_step_data.append(data)

        if n_lat is None or n_lon is None:
            raise RuntimeError("Could not infer 2D mask shape from source files.")

        all_data.append(np.stack(time_step_data, axis=0))

    return np.ascontiguousarray(np.stack(all_data, axis=0).astype(np.float32))


def write_output(data: np.ndarray, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output_file, "w") as ds:
        ds.createDimension("time", data.shape[0])
        ds.createDimension("height", data.shape[1])
        ds.createDimension("lat", data.shape[2])
        ds.createDimension("lon", data.shape[3])
        out = ds.createVariable("source_mask", "f4", ("time", "height", "lat", "lon"))
        out[:] = data


def main() -> None:
    parser_obj = argparse.ArgumentParser(
        description="Assemble TIANGOU snapshots into one source-mask cube."
    )
    parser_obj.add_argument("input_dir", type=Path, help="Directory containing TIANGOU snapshots")
    parser_obj.add_argument(
        "--output",
        type=Path,
        default=Path("TIANGOU_SourceMask.nc"),
        help="Path of the merged NetCDF file",
    )
    args = parser_obj.parse_args()

    data = build_stack(args.input_dir)
    write_output(data, args.output)
    print(
        f"Wrote {args.output} with shape "
        f"(time={data.shape[0]}, height={data.shape[1]}, lat={data.shape[2]}, lon={data.shape[3]})"
    )


if __name__ == "__main__":
    main()
