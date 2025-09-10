from __future__ import annotations

import os
import time
from pathlib import Path
import pytest


@pytest.mark.compliance
def test_retention_policy_simulation(tmp_path: Path):
    p = tmp_path / "old_file.bin"
    p.write_bytes(b"x")
    old = time.time() - 60 * 60 * 24 * 400  # 400 days ago
    os.utime(p, (old, old))
    # Simulate retention of 365 days
    cutoff = time.time() - 60 * 60 * 24 * 365
    expired = [f for f in tmp_path.iterdir() if f.stat().st_mtime < cutoff]
    assert p in expired

