from __future__ import annotations

import os
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import app


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_package_import_and_api_roundtrip(tmp_path: Path, monkeypatch):
    # Ensure isolated state for the API
    monkeypatch.setenv("HOLO_ROOT", str(tmp_path))
    monkeypatch.setenv("HLOG_DATA_DIR", str(tmp_path / "holographic_memory"))
    monkeypatch.setenv("HOLO_FALLBACK_BASE64", "true")
    # Verify package API imports
    import holographic_memory as hm  # type: ignore

    assert hasattr(hm, "HolographicMemory")
    H = hm.HolographicMemory(root=tmp_path)
    # Store/retrieve through package
    payload = b"production-validation-payload"
    did = H.store(payload, filename="pv.bin")
    back = H.retrieve(did)
    assert sha256(back) == sha256(payload)

    # Exercise API endpoints via TestClient in the same process
    c = TestClient(app)
    files = {"file": ("pv.txt", payload, "text/plain")}
    r = c.post("/store", files=files, headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]
    d = c.get(f"/download/{doc_id}", headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
    assert d.status_code == 200
    assert sha256(d.content) == sha256(payload)

    # Telemetry exists and has overall metrics
    t = c.get("/telemetry", headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
    assert t.status_code == 200
    tj = t.json()
    assert "telemetry" in tj and "overall" in tj["telemetry"]

