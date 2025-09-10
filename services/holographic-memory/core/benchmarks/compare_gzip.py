from __future__ import annotations

import gzip
import os
import time
from pathlib import Path

from holographicfs.memory import mount


def size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> None:
    # Prepare sample files
    root = Path.cwd() / "holo_bench"
    root.mkdir(exist_ok=True)
    small = root / "small.txt"
    medium = root / "medium.txt"
    large = root / "large.bin"
    small.write_text("hello world " * 256, encoding="utf-8")
    medium.write_text("Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n" * 4096, encoding="utf-8")
    large.write_bytes(os.urandom(2 * 1024 * 1024))  # 2MB

    fs = mount(root, grid_size=64)

    # Holographic store timing
    t0 = time.time(); d1 = fs.store(small); t1 = time.time()
    d2 = fs.store(medium); t2 = time.time()
    d3 = fs.store(large); t3 = time.time()

    # Gzip sizes
    gz_small = root / (small.name + ".gz"); gz_small.write_bytes(gzip.compress(small.read_bytes()))
    gz_medium = root / (medium.name + ".gz"); gz_medium.write_bytes(gzip.compress(medium.read_bytes()))
    gz_large = root / (large.name + ".gz"); gz_large.write_bytes(gzip.compress(large.read_bytes()))

    stats = fs.stats()
    total_raw = small.stat().st_size + medium.stat().st_size + large.stat().st_size
    total_gz = size(gz_small) + size(gz_medium) + size(gz_large)
    total_holo = stats.get("holo_bytes", 0)

    print("Holographic store latency (ms):", round((t1 - t0) * 1000, 2), round((t2 - t1) * 1000, 2), round((t3 - t2) * 1000, 2))
    print("Total raw bytes:", total_raw)
    print("Total gzip bytes:", total_gz)
    print("Total holographic state bytes:", total_holo)
    if total_holo:
        print("Compression vs raw (x):", round(total_raw / total_holo, 2))
    if total_gz:
        print("Compression vs gzip (x):", round(total_gz / total_holo, 2))


if __name__ == "__main__":
    main()

