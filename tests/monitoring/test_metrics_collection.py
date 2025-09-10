from __future__ import annotations

import pytest


@pytest.mark.integration
def test_prometheus_metrics_endpoint_available(api_client):  # type: ignore
    r = api_client.get("/metrics")
    assert r.status_code == 200
    assert b"holographic_store_requests_total" in r.content

