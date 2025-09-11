from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed_section() -> Iterator[float]:
    start = time.perf_counter()
    yield start
    end = time.perf_counter()
    # Returned indirectly via caller measuring end-start

