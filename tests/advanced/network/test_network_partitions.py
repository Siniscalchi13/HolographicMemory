from __future__ import annotations

import httpx
import pytest


@pytest.mark.network
def test_network_partition_simulation(monkeypatch):
    # Simulate partition by forcing ConnectError
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("partition")

    monkeypatch.setattr(httpx, "get", _raise)
    with pytest.raises(httpx.ConnectError):
        httpx.get("http://example.com")

