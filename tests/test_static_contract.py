from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from astropy.io import fits
from netCDF4 import Dataset
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StaticContractTest(unittest.TestCase):
    def test_fism_event_token(self) -> None:
        module = load_module("download_fism2_event", ROOT / "scripts" / "download_fism2_event.py")
        _, ydoy = module.event_tokens("2024-04-08")
        self.assertEqual(ydoy, "2024099")

    def test_tiangou_source_count(self) -> None:
        module = load_module(
            "make_tiangou_23band_mask",
            ROOT / "pipeline" / "make_tiangou_23band_mask.py",
        )
        self.assertEqual(module.MODEL_NAME, "TIANGOU Eclipse Model")
        self.assertEqual(
            module.MODEL_FULL_NAME,
            "Topography-aware Irradiance And Nonuniform-Geometry Occultation Utility",
        )
        sources = list(module.SOURCE_CONFIGS) + [module.AIA1600_SOURCE]
        self.assertEqual(len(sources), 9)
        self.assertEqual(sources[5].source_id, "suvi_284")

        source_ids = [source.source_id for source in sources]
        weights = module.build_source_weights(np.array([40.0]), source_ids)
        self.assertEqual(float(weights[0, source_ids.index("aia_1600")]), 1.0)
        self.assertEqual(int(np.count_nonzero(weights[0])), 1)

    def test_dem_manifest_sizes(self) -> None:
        module = load_module("download_lunar_dem", ROOT / "scripts" / "download_lunar_dem.py")
        image_bytes = sum(item.size_bytes for item in module.PRODUCTS if item.relative_path.endswith(".img"))
        self.assertEqual(image_bytes, 3_312_896_000)

    def test_vertical_grid_has_70_levels(self) -> None:
        grid = ROOT / "pipeline" / "vertical_grids" / "waccm6_70_heights_km.txt"
        values = [
            float(line)
            for line in grid.read_text(encoding="ascii").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertEqual(len(values), 70)
        self.assertGreater(values[0], values[-1])

    def test_native_solar_reader_preserves_pixels_and_geometry(self) -> None:
        module = load_module("solar_image_test", ROOT / "tiangou" / "solar_image.py")
        source = np.arange(15, dtype=np.float32).reshape(3, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aia.lev1.193A_2024_04_08T18_00_00.00Z.image_lev1.fits"
            image_hdu = fits.ImageHDU(source)
            image_hdu.header["DATE-OBS"] = "2024-04-08T18:00:00Z"
            image_hdu.header["CRPIX1"] = 2.25
            image_hdu.header["CRPIX2"] = 1.25
            image_hdu.header["CDELT1"] = 0.6
            fits.HDUList([fits.PrimaryHDU(), image_hdu]).writeto(path)
            loaded = module.load_solar_image(
                directory,
                instrument="aia",
                wavelength=193,
                target_time=module.datetime(2024, 4, 8, 18, 0),
            )

        np.testing.assert_array_equal(loaded.data, source)
        self.assertEqual(loaded.data.shape, source.shape)
        self.assertEqual((loaded.center_column, loaded.center_row), (2.25, 1.25))

    def test_native_sldem_reader_visits_every_pixel_in_domain(self) -> None:
        module = load_module("lunar_dem_native_test", ROOT / "tiangou" / "lunar_dem.py")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "tiny.img"
            label_path = root / "tiny.lbl"
            np.arange(6, dtype="<f4").tofile(image_path)
            label_path.write_text(
                "\n".join(
                    (
                        "LINES = 2",
                        "LINE_SAMPLES = 3",
                        "SAMPLE_TYPE = PC_REAL",
                        "SAMPLE_BITS = 32",
                        "MAP_RESOLUTION = 1",
                        "MINIMUM_LATITUDE = -1",
                        "MAXIMUM_LATITUDE = 1",
                        "WESTERNMOST_LONGITUDE = 0",
                        "OFFSET = 1737.4",
                    )
                )
                + "\n",
                encoding="ascii",
            )

            blocks = list(
                module.iter_sldem_point_blocks(
                    image_path,
                    label_path,
                    row_block_size=1,
                    col_block_size=2,
                )
            )

        self.assertEqual(sum(block.shape[0] for block in blocks), 6)

    def test_source_cube_assembly_requires_complete_snapshots(self) -> None:
        module = load_module(
            "assemble_source_cube_test",
            ROOT / "pipeline" / "assemble_source_cube.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for time_index, stamp in enumerate(("20240408180000", "20240408180500")):
                time_values = []
                for height_index, height in enumerate((100.0, 90.0)):
                    values = np.full((2, 3), time_index * 10 + height_index, dtype=np.float32)
                    path = root / f"{stamp}_{height:.2f}km_193.nc"
                    with Dataset(path, "w") as dataset:
                        dataset.createDimension("glat", 2)
                        dataset.createDimension("glon", 3)
                        variable = dataset.createVariable("transmission", "f4", ("glat", "glon"))
                        variable[:] = values
                    time_values.append(values)
                expected.append(time_values)

            assembled = module.build_stack(root)
            np.testing.assert_array_equal(assembled, np.asarray(expected, dtype=np.float32))

            (root / "20240408180500_90.00km_193.nc").unlink()
            with self.assertRaises(FileNotFoundError):
                module.build_stack(root)


if __name__ == "__main__":
    unittest.main()
