from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from .utils import upload_file, download_doc


@pytest.mark.skipif(os.getenv("HOLO_RUN_SLOW", "0") != "1", reason="slow/performance disabled; set HOLO_RUN_SLOW=1")
def test_large_file_upload_performance(base_url: str, tmp_path: Path):
    size_mb = int(os.getenv("HOLO_LARGE_MB", "12"))
    data = os.urandom(size_mb * 1024 * 1024)
    f = tmp_path / "large.bin"
    f.write_bytes(data)
    t0 = time.time()
    doc_id, _ = upload_file(base_url, f)
    t1 = time.time()
    # rough target: < 5s for ~12MB on local dev
    assert (t1 - t0) < float(os.getenv("HOLO_UPLOAD_MAX_S", "5.0"))
    # Verify download throughput
    t2 = time.time(); got = download_doc(base_url, doc_id); t3 = time.time()
    assert got[:64] == data[:64] and len(got) == len(data)
    # rough target: < 5s
    assert (t3 - t2) < float(os.getenv("HOLO_DOWNLOAD_MAX_S", "5.0"))

