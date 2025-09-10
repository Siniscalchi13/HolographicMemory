"""Global pytest configuration and shared fixtures for enterprise tests.

This file sets up environment isolation, shared markers, and base fixtures
including a temporary holographic data directory, a FastAPI TestClient for
the SOA API (when available), and utilities for timeouts and retries.
"""
from __future__ import annotations

import os
import typing as _t
from pathlib import Path

import pytest

# Enable pytest plugin for structured logging and HTML generation
pytest_plugins = ("tests.logging.plugin",)


@pytest.fixture(autouse=True)
def _isolate_hlog_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate holographic data directory per test run.

    Sets env vars used across services to point to a temporary directory to
    avoid interfering with developer or CI machines.
    """
    data_dir = tmp_path / "holographic_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HLOG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HOLOGRAPHIC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HOLOGRAPHIC_USE_GPU", "false")
    # Deterministic thresholds for router decisions
    monkeypatch.setenv("HOLO_MICRO_THRESHOLD", "256")
    monkeypatch.setenv("HOLO_MICRO_K8_MAX", "1024")


class _FakeMemory:
    def __init__(self, state_dir, grid_size: int = 64, use_gpu: bool = False) -> None:
        self.state = {}
        self.grid_size = int(grid_size)
        self.use_gpu = bool(use_gpu)
        self.backend = object()

    def store_bytes(self, doc_id: str, content: bytes):
        self.state[str(doc_id)] = bytes(content)
        return {"ok": True, "encoded_data": content[:16]}

    def retrieve_bytes(self, doc_id: str) -> bytes:
        return self.state.get(str(doc_id), b"")


@pytest.fixture(autouse=True, scope="session")
def _patch_orchestrator_memory() -> None:
    try:
        import services.orchestrator.orchestrator as orch
        # Direct patching for session scope
        original = getattr(orch, "HolographicMemory", None)
        orch.HolographicMemory = _FakeMemory
        yield
        # Restore original if it existed
        if original is not None:
            orch.HolographicMemory = original
    except Exception:
        yield

@pytest.fixture(scope="session")
def test_config() -> dict[str, _t.Any]:
    """Load the shared test configuration YAML if present."""
    import yaml  # type: ignore

    cfg_path = Path("tests/config/test_config.yaml")
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return _t.cast(dict[str, _t.Any], yaml.safe_load(f) or {})


@pytest.fixture(scope="session")
def event_loop():  # type: ignore[override]
    """Use a single event loop for async tests across the session."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def fastapi_app():
    """Provide the SOA FastAPI app if importable.

    Falls back gracefully if the API app is not available (tests using this
    fixture should mark themselves with `contract` or `integration`).
    """
    try:
        # Prefer the SOA app with orchestrator dep injection (underscore variant)
        from services.holographic_memory.api.app_soa import app  # type: ignore
        return app
    except Exception:
        pass
    try:
        # Fallback: load module by path (hyphenated folder name)
        import runpy
        mod = runpy.run_path("services/holographic-memory/api/app_soa.py")
        app = mod.get("app")
        if app is None:
            raise RuntimeError("app not defined in app_soa.py")
        return app
    except Exception as exc:
        pytest.skip(f"FastAPI SOA app not importable: {exc}")


@pytest.fixture()
def api_client(fastapi_app):  # type: ignore[override]
    """Return a TestClient for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # pragma: no cover - dependency not present
        pytest.skip(f"fastapi.testclient unavailable: {exc}")
    return TestClient(fastapi_app)


def pytest_configure(config: pytest.Config) -> None:
    # Ensure markers are registered to avoid warnings when selecting
    for m in (
        "unit",
        "integration",
        "contract",
        "e2e",
        "performance",
        "security",
        "chaos",
        "gpu",
    ):
        config.addinivalue_line("markers", f"{m}:")
