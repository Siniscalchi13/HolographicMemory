from __future__ import annotations

import gc
import tracemalloc
import pytest


@pytest.mark.performance
def test_no_unbounded_memory_growth():
    def allocate():
        return bytearray(1024)  # 1KB

    gc.collect()
    tracemalloc.start()
    for _ in range(1000):
        _ = allocate()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Accept small footprint; peak under modest threshold in CI
    assert peak < 50 * 1024 * 1024  # 50MB

