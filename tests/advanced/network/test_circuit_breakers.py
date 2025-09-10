from __future__ import annotations

import time
import pytest


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_seconds: float = 0.05):
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.open_until = 0.0
        self.reset_seconds = reset_seconds

    def call(self, func):
        now = time.time()
        if now < self.open_until:
            raise RuntimeError("circuit open")
        try:
            res = func()
            self.failures = 0
            return res
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.open_until = now + self.reset_seconds
            raise


@pytest.mark.network
def test_circuit_opens_on_consecutive_failures():
    cb = CircuitBreaker(failure_threshold=2, reset_seconds=0.05)

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cb.call(boom)
    with pytest.raises(RuntimeError):
        cb.call(boom)
    # Now circuit should be open
    with pytest.raises(RuntimeError, match="circuit open"):
        cb.call(lambda: 1)
    time.sleep(0.06)
    # After reset, calls can pass
    assert cb.call(lambda: 2) == 2

