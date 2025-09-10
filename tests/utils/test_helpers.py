from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Callable, Any


def approx_ratio(numer: int, denom: int, target: float, tol: float = 0.05) -> bool:
    if denom == 0:
        return False
    return abs((numer / float(denom)) - target) <= tol


@contextmanager
def time_block() -> Iterator[Callable[[], float]]:
    """Context manager to measure elapsed wall time in seconds.

    Usage:
        with time_block() as elapsed:
            ...
        assert elapsed() < 0.1
    """
    t0 = time.perf_counter()
    yield lambda: (time.perf_counter() - t0)

