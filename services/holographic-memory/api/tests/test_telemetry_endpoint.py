import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import app


def test_telemetry_endpoint_basic():
    with tempfile.TemporaryDirectory() as td:
        os.environ["HOLO_ROOT"] = td
        os.environ["HLOG_DATA_DIR"] = str(Path(td) / "holographic_memory")
        client = TestClient(app)
        # store a couple files
        files = {"file": ("a.txt", b"some data" * 10, "text/plain")}
        r = client.post("/store", files=files, headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
        assert r.status_code == 200
        files = {"file": ("b.txt", b"more data" * 20, "text/plain")}
        r = client.post("/store", files=files, headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
        assert r.status_code == 200
        # telemetry endpoint
        t = client.get("/telemetry", headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
        assert t.status_code == 200
        js = t.json()
        assert "telemetry" in js and "suggested_dimensions" in js
        assert "overall" in js["telemetry"]

