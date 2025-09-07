from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple


@dataclass
class Timing:
    count: int
    p50_ms: float
    p95_ms: float
    avg_ms: float


class MetricsService:
    """
    Lightweight performance metrics: latency histograms and simple throughput.
    """

    @staticmethod
    def time_many(fn: Callable[[], None], n: int = 10) -> Timing:
        samples: List[float] = []
        for _ in range(max(1, n)):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)
        samples.sort()
        p50 = statistics.median(samples)
        p95 = samples[max(0, int(len(samples) * 0.95) - 1)]
        avg = sum(samples) / len(samples)
        return Timing(count=n, p50_ms=p50, p95_ms=p95, avg_ms=avg)

