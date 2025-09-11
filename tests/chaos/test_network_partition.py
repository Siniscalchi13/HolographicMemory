from __future__ import annotations

import pytest

from tests.infrastructure.chaos_injector import inject_exception


class _T:
    def ping(self) -> str:
        return "pong"


@pytest.mark.chaos
def test_inject_exception_context_manager():
    t = _T()
    assert t.ping() == "pong"
    with inject_exception(t, "ping", RuntimeError("partition")):
        with pytest.raises(RuntimeError):
            t.ping()
    # restored
    assert t.ping() == "pong"

