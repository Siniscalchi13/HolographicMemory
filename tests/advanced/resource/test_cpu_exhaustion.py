from __future__ import annotations

import time
import pytest


@pytest.mark.performance
def test_cpu_spike_short_duration():
    # Busy loop for a very short time to avoid overloading CI
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.01:
        pass
    assert True

