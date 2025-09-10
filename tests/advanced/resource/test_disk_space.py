from __future__ import annotations

from pathlib import Path
import os
import pytest


@pytest.mark.performance
def test_write_small_file_ok(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(os.urandom(1024 * 1024))  # 1MB
    assert p.exists() and p.stat().st_size == 1024 * 1024

