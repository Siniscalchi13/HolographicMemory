from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class PerfSample:
    latency_s: float
    rss_delta_kb: int
    peak_kb: int


def measure(func: Callable[[], Any], warmups: int = 1) -> PerfSample:
    """Run a callable and measure latency and memory deltas.

    Returns a PerfSample with elapsed time and memory footprint estimate
    using tracemalloc. Intended for deterministic micro-benchmarks in unit
    and integration tests; full load tests live under performance/.
    """
    gc.collect()
    tracemalloc.start()
    for _ in range(max(0, warmups)):
        func()
    t0 = time.perf_counter()
    func()
    dt = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Convert to kilobytes
    return PerfSample(latency_s=dt, rss_delta_kb=int(cur / 1024), peak_kb=int(peak / 1024))

