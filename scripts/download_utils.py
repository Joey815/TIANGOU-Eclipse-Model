from __future__ import annotations

from pathlib import Path
import shutil

import requests


def download_resumable(
    url: str,
    destination: str | Path,
    expected_size: int | None = None,
    overwrite: bool = False,
    timeout: tuple[int, int] = (30, 600),
) -> Path:
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        size = destination.stat().st_size
        if not overwrite and (expected_size is None or size == expected_size):
            print(f"cached {destination} ({size} bytes)")
            return destination
        if not overwrite:
            raise RuntimeError(
                f"Existing file has unexpected size: {destination} ({size} bytes)"
            )
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() and not overwrite else 0
    if overwrite and partial.exists():
        partial.unlink()

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    print(f"fetch {url}")
    print(f"  -> {destination} (resume={offset})")
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        if offset and not append:
            offset = 0
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)

    final_size = partial.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise RuntimeError(
            f"Download size mismatch for {destination.name}: "
            f"expected {expected_size}, got {final_size}"
        )
    shutil.move(str(partial), str(destination))
    return destination
