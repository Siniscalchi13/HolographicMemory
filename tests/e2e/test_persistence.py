from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

from .utils import upload_file, download_doc


@pytest.mark.skipif(os.getenv("HOLO_E2E_ALLOW_DOCKERCTL", "0") != "1", reason="restart requires docker control")
def test_persistence_across_restart(base_url: str, tmp_path: Path):
    from .conftest import restart_api_if_allowed

    data = b"persistence check bytes"
    f = tmp_path / "persist.bin"
    f.write_bytes(data)
    doc_id, _ = upload_file(base_url, f)

    restart_api_if_allowed(base_url)

    # Verify list still contains the file and download works
    rows = requests.get(f"{base_url}/list", timeout=10).json().get("results", [])
    assert any(r.get("doc_id") == doc_id for r in rows)
    got = download_doc(base_url, doc_id)
    assert got == data

