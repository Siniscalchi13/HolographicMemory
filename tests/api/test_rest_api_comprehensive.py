from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("path", [
    "/healthz",
    "/capabilities",
    "/stats",
    "/metrics",
    "/api/real-metrics",
    "/api/system-status",
    "/api/gpu-status",
    "/api/memory-usage",
    "/api/performance",
])
def test_api_endpoints_available(api_client, path: str):  # type: ignore
    r = api_client.get(path)
    assert r.status_code == 200
    assert isinstance(r.json() if r.headers.get("content-type", "").endswith("json") else {}, (dict,)) or r.content


@pytest.mark.integration
def test_api_rebalance_endpoint(api_client):  # type: ignore
    r = api_client.post("/rebalance", json={"force": True})
    assert r.status_code == 200
    j = r.json()
    assert j.get("success") is True
    assert "rebalancing_result" in j

