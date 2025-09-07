from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Generator

import pytest
import requests


def _base_url() -> str:
    return os.getenv("HOLO_BASE_URL", "http://localhost:8000").rstrip("/")


def wait_for_health(base: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/healthz", timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def base_url() -> str:
    base = _base_url()
    if not wait_for_health(base, timeout=30):
        pytest.skip(f"API not healthy at {base}")
    return base


@pytest.fixture()
def tmp_file(tmp_path: Path) -> Generator[Path, None, None]:
    p = tmp_path / "sample.txt"
    p.write_text("hello holographic world", encoding="utf-8")
    yield p


def restart_api_if_allowed(base: str) -> None:
    if os.getenv("HOLO_E2E_ALLOW_DOCKERCTL", "0") != "1":
        pytest.skip("docker control disabled; set HOLO_E2E_ALLOW_DOCKERCTL=1 to enable restart tests")
    # Try graceful shutdown via API
    try:
        requests.post(f"{base}/shutdown", timeout=2)
    except Exception:
        pass
    # Bring containers back up
    os.system("docker compose up -d >/dev/null 2>&1")
    assert wait_for_health(base, 40), "API did not become healthy after restart"

