import os
import tempfile
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import app


def test_vault_secret_routes_to_micro_and_random_id():
    with tempfile.TemporaryDirectory() as td:
        os.environ["HOLO_ROOT"] = td
        os.environ["HLOG_DATA_DIR"] = str(Path(td) / "holographic_memory")
        os.environ["HOLO_MICRO_THRESHOLD"] = "256"
        client = TestClient(app)
        content = b"API_KEY=supersecret"
        files = {"file": (".env", content, "text/plain")}
        r = client.post("/store", files=files, headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
        assert r.status_code == 200, r.text
        doc_id = r.json()["doc_id"]
        # doc_id should not equal sha256(content)
        assert doc_id != hashlib.sha256(content).hexdigest()
        # Sidecar should not exist for vault
        patterns = Path(os.environ["HLOG_DATA_DIR"]) / "patterns"
        p = patterns / ".env.hwp.json"
        assert not p.exists()

