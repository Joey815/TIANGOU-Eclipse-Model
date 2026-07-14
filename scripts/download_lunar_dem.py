#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from download_utils import download_resumable


SLDEM_BASE = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/sldem2015/global/float_img"
)
LDEM_BASE = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/polar/float_img"
)


@dataclass(frozen=True)
class Product:
    group: str
    relative_path: str
    url: str
    size_bytes: int


PRODUCTS = (
    Product(
        "global",
        "sldem2015_128_60s_60n_000_360_float.img",
        f"{SLDEM_BASE}/sldem2015_128_60s_60n_000_360_float.img",
        2_831_155_200,
    ),
    Product(
        "global",
        "sldem2015_128_60s_60n_000_360_float.lbl",
        f"{SLDEM_BASE}/sldem2015_128_60s_60n_000_360_float.lbl",
        4_588,
    ),
    Product("polar60", "polar/ldem_60n_240m_float.img", f"{LDEM_BASE}/ldem_60n_240m_float.img", 240_870_400),
    Product("polar60", "polar/ldem_60n_240m_float.lbl", f"{LDEM_BASE}/ldem_60n_240m_float.lbl", 4_809),
    Product("polar60", "polar/ldem_60s_240m_float.img", f"{LDEM_BASE}/ldem_60s_240m_float.img", 240_870_400),
    Product("polar60", "polar/ldem_60s_240m_float.lbl", f"{LDEM_BASE}/ldem_60s_240m_float.lbl", 4_877),
)


def main() -> None:
    parser = ArgumentParser(description="Download the lunar DEM inputs used by the true-limb workflow.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("global", "polar60"),
        default=("global", "polar60"),
    )
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = set(args.groups)
    for product in PRODUCTS:
        if product.group not in selected:
            continue
        if args.labels_only and not product.relative_path.endswith(".lbl"):
            continue
        download_resumable(
            product.url,
            args.output_dir / product.relative_path,
            expected_size=product.size_bytes,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
