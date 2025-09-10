from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_dashboard_static_index_exists():
    idx = Path("services/dashboard/web/index.html")
    assert idx.exists(), "Dashboard index.html should exist"

