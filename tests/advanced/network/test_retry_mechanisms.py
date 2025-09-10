from __future__ import annotations

import pytest
from tenacity import retry, stop_after_attempt, wait_fixed


_calls = {"n": 0}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
def flaky_call() -> int:
    _calls["n"] += 1
    if _calls["n"] < 3:
        raise RuntimeError("transient")
    return 42


@pytest.mark.network
def test_retry_succeeds_on_third_attempt():
    _calls["n"] = 0
    assert flaky_call() == 42
    assert _calls["n"] == 3

