from __future__ import annotations

import io
import pytest


@pytest.mark.e2e
def test_end_to_end_store_search_download(api_client):  # type: ignore
    # Store a file
    files = {"file": ("manual.txt", b"quantum holography notes", "text/plain")}
    r = api_client.post("/store", files=files)
    assert r.status_code == 200
    doc_id = r.json()["doc_id"]
    # Search (placeholder — service returns []) but should respond
    r = api_client.post("/search", json={"query": "quantum", "limit": 5})
    assert r.status_code == 200
    # Download
    r = api_client.get(f"/download/{doc_id}")
    assert r.status_code == 200 and b"holography" in r.content

