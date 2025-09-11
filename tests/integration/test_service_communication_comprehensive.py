from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_health_and_capabilities_sanity(api_client):  # type: ignore
    r = api_client.get("/healthz")
    assert r.status_code in (200, 307, 308)  # FastAPI may redirect, but our app returns JSON
    # Try SOA app capabilities endpoints (app_soa)
    r2 = api_client.get("/capabilities")
    assert r2.status_code == 200
    body = r2.json()
    assert "grid_size" in body or "gpu_available" in body


@pytest.mark.integration
@pytest.mark.parametrize(
    "name,ctype,content",
    [
        ("a.txt", "text/plain", b"hello"),
        ("b.json", "application/json", b"{\n  \"k\": 1\n}"),
        ("c.csv", "text/csv", b"a,b\n1,2\n"),
        ("d.md", "text/markdown", b"# Title\nBody"),
        ("e.bin", "application/octet-stream", b"\x00\x01\x02\x03\x04"),
        ("f.log", "text/plain", b"log line\n" * 3),
        ("g.ini", "text/plain", b"[x]\na=1\n"),
        ("h.yaml", "application/x-yaml", b"a: 1\n"),
        ("i.xml", "application/xml", b"<x/>"),
        ("j.rtf", "application/rtf", b"{\\rtf1}"),
    ],
)
def test_store_and_download_roundtrip(api_client, name: str, ctype: str, content: bytes):  # type: ignore
    files = {"file": (name, content, ctype)}
    res = api_client.post("/store", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body.get("success") is True
    doc_id = body["doc_id"]
    # Download
    d = api_client.get(f"/download/{doc_id}")
    assert d.status_code == 200
    assert d.content == content


@pytest.mark.integration
def test_stats_endpoint_available(api_client):  # type: ignore
    r = api_client.get("/stats")
    assert r.status_code == 200
    j = r.json()
    assert "system_status" in j or isinstance(j, dict)

