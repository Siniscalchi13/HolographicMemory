from __future__ import annotations

import psutil
import pytest


@pytest.mark.performance
def test_capture_cpu_and_mem():
    cpu = psutil.cpu_percent(interval=0.05)
    mem = psutil.virtual_memory().percent
    assert 0.0 <= cpu <= 100.0
    assert 0.0 <= mem <= 100.0

