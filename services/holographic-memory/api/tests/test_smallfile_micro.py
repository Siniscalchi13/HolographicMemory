import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import app


def test_small_file_micro_header():
    with tempfile.TemporaryDirectory() as td:
        os.environ["HOLO_ROOT"] = td
        os.environ["HLOG_DATA_DIR"] = str(Path(td) / "holographic_memory")
        os.environ["HOLO_MICRO_THRESHOLD"] = "256"
        client = TestClient(app)
        content = b"tiny-config=true"  # < 256 bytes
        files = {"file": ("tiny.txt", content, "text/plain")}
        r = client.post("/store", files=files, headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
        assert r.status_code == 200, r.text
        # Validate file on disk is micro-sized
        patterns = Path(os.environ["HLOG_DATA_DIR"]) / "patterns"
        p = patterns / "tiny.txt.hwp"
        assert p.exists()
        sz = p.stat().st_size
        assert sz <= 32

