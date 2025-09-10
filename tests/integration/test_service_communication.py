from __future__ import annotations

import io
import pytest


@pytest.mark.integration
def test_health_and_capabilities(api_client):  # type: ignore
    r = api_client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "healthy"
    r = api_client.get("/capabilities")
    assert r.status_code == 200
    data = r.json()
    assert "layer_dimensions" in data and "services" in data


@pytest.mark.integration
def test_store_and_download_flow(api_client):  # type: ignore
    # Upload/store
    files = {"file": ("hello.txt", b"hello world", "text/plain")}
    r = api_client.post("/store", files=files)
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]
    # Download
    r = api_client.get(f"/download/{doc_id}")
    assert r.status_code == 200
    assert b"hello world" in r.content

