import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import app


def test_large_file_prefers_v4_and_is_small_on_disk():
    with tempfile.TemporaryDirectory() as td:
        os.environ["HOLO_ROOT"] = td
        os.environ["HLOG_DATA_DIR"] = str(Path(td) / "holographic_memory")
        # thresholds to force v4 for large
        os.environ["HOLO_MICRO_THRESHOLD"] = "256"
        os.environ["HOLO_MICRO_K8_MAX"] = "1024"
        client = TestClient(app)
        content = (b"This is a large payload for holographic testing.\n" * 20000)  # ~1.1MB
        files = {"file": ("large.txt", content, "text/plain")}
        r = client.post("/store", files=files, headers={"x-api-key": os.getenv("HOLO_API_KEY", "")})
        assert r.status_code == 200, r.text
        patterns = Path(os.environ["HLOG_DATA_DIR"]) / "patterns"
        p = patterns / "large.txt.hwp"
        assert p.exists()
        sz = p.stat().st_size
        assert sz < 2000  # expect small binary artifact

