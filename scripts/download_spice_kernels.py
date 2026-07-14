#!/usr/bin/env python3.11

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiangou.kernels import TIANGOU_KERNEL_SET, ensure_kernel_set


def main() -> None:
    parser = ArgumentParser(description="Download the NAIF kernels used by TIANGOU.")
    parser.add_argument(
        "--kernel-dir",
        default=str(ROOT / "data" / "kernels"),
        help="Target kernel directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing kernels")
    args = parser.parse_args()

    results = ensure_kernel_set(
        target_dir=args.kernel_dir,
        kernel_names=TIANGOU_KERNEL_SET,
        overwrite=args.overwrite,
    )

    print(f"Kernel directory: {Path(args.kernel_dir).expanduser().resolve()}")
    total_bytes = 0
    for result in results:
        total_bytes += result.size_bytes
        status = "downloaded" if result.downloaded else "cached"
        print(f"{result.name:30s} {status:10s} {result.size_bytes / (1024**2):8.2f} MiB  {result.source_url}")
    print(f"Total: {total_bytes / (1024**2):.2f} MiB")


if __name__ == "__main__":
    main()
