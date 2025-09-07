from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from holographicfs.memory import mount


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="holo_test_"))
    try:
        root = tmp / "store"
        root.mkdir(parents=True, exist_ok=True)
        fs = mount(root, grid_size=32)

        # Create a sample file
        sample = root / "demo.txt"
        sample.write_text("this is a holographic test file", encoding="utf-8")

        # Store
        doc_id = fs.store(sample)
        print(f"STORED\t{doc_id}\t{sample}")

        # Search by name (index-backed)
        idx = fs.search_index("demo")
        assert idx, "index search returned no results"
        print("SEARCH_IDX\t", idx[0])

        # Recall (fallback to original if HM chunks unavailable)
        out = root / "recalled.bin"
        recalled = fs.recall(doc_id, out=out)
        assert recalled.exists() and recalled.stat().st_size > 0
        assert recalled.read_bytes() == sample.read_bytes()
        print(f"RECALLED\t{recalled}")

        # Stats
        s = fs.stats()
        assert s.get("files_indexed", 0) >= 1
        print("STATS\t", s)

        print("OK: end-to-end holographic system test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

