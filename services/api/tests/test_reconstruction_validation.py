import os
import tempfile
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import app


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_reconstruction_across_sizes():
    with tempfile.TemporaryDirectory() as td:
        os.environ["HOLO_ROOT"] = td
        os.environ["HLOG_DATA_DIR"] = str(Path(td) / "holographic_memory")
        # Enable sidecar to ensure reconstruction when 3D backend is unavailable
        os.environ["HOLO_FALLBACK_BASE64"] = "true"
        os.environ["HOLO_MICRO_THRESHOLD"] = "256"
        os.environ["HOLO_MICRO_K8_MAX"] = "1024"
        client = TestClient(app)

        payloads = [
            ("micro.txt", b"tiny conf=true", "text/plain"),
            ("microk8.txt", b"A" * 900, "text/plain"),
            ("v4.txt", (b"This is a large payload for holographic testing.\n" * 5000), "text/plain"),
        ]
        for name, data, ctype in payloads:
            files = {"file": (name, data, ctype)}
            r = client.post("/store", files=files, headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
            assert r.status_code == 200, r.text
            doc_id = r.json()["doc_id"]
            d = client.get(f"/download/{doc_id}", headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
            assert d.status_code == 200
            assert sha256(d.content) == sha256(data)

