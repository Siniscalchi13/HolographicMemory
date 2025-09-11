from __future__ import annotations

import io
from pathlib import Path

import pytest


@pytest.mark.integration
def test_end_to_end_store_retrieve_multiple(api_client):  # type: ignore
    # Upload several files and ensure list/download works
    payloads = [
        ("a.txt", b"alpha"),
        ("b.txt", b"beta"),
        ("c.txt", b"gamma"),
        ("d.json", b"{\"x\":1}"),
        ("e.bin", b"\x00\x01\x02\x03\x04\x05"),
    ]
    ids: list[str] = []
    for name, data in payloads:
        res = api_client.post("/store", files={"file": (name, data, "application/octet-stream")})
        assert res.status_code == 200
        ids.append(res.json()["doc_id"])
    # Download back
    for (name, data), doc_id in zip(payloads, ids):
        d = api_client.get(f"/download/{doc_id}")
        assert d.status_code == 200
        assert d.content == data

