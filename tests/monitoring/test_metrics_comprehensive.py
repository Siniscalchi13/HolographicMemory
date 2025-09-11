from __future__ import annotations

import pytest


@pytest.mark.monitoring
def test_prometheus_metrics_and_counters(api_client):  # type: ignore
    # Access metrics endpoint and simple store/search counters indirectly
    m = api_client.get("/metrics")
    assert m.status_code == 200
    assert b"holographic_store_requests_total" in m.content or b"holographic_search_requests_total" in m.content

