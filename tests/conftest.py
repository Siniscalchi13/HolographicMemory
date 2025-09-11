"""Global pytest configuration and shared fixtures for enterprise tests.

This file sets up environment isolation, shared markers, and base fixtures
including a temporary holographic data directory, a FastAPI TestClient for
the SOA API (when available), and utilities for timeouts and retries.
"""
from __future__ import annotations

import os
import sys
import typing as _t
from pathlib import Path

import pytest

# Enable pytest plugin for structured logging and HTML generation
pytest_plugins = ("tests.logging.plugin",)


@pytest.fixture(autouse=True)
def _isolate_hlog_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate holographic data directory per test run and enable real GPU-backed memory if available."""
    data_dir = tmp_path / "holographic_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HLOG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HOLOGRAPHIC_DATA_DIR", str(data_dir))
    # Enable GPU usage by default; real engine will gracefully fall back if unavailable
    monkeypatch.setenv("HOLOGRAPHIC_USE_GPU", "true")
    # Deterministic thresholds for router decisions
    monkeypatch.setenv("HOLO_MICRO_THRESHOLD", "256")
    monkeypatch.setenv("HOLO_MICRO_K8_MAX", "1024")


@pytest.fixture(autouse=True, scope="session")
def _enable_native_backends() -> None:
    """Ensure native GPU/CPU backends are importable by extending sys.path to bundled libs.

    Looks for prebuilt shared objects under services/holographic-memory/core/native/holographic/* and appends
    their directories to sys.path so imports like `import holographic_gpu` succeed without system installation.
    """
    root = Path.cwd() / "services" / "holographic-memory" / "core" / "native" / "holographic"
    candidates = []
    # Prioritize build directory (Python 3.13 versions) over lib.* directories (Python 3.12 versions)
    for sub in ("build",):
        p = root / sub
        if p.exists():
            candidates.append(p)
    # Also search lib.* directories (e.g., lib.macosx-metal, lib.linux-cuda) as fallback
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith("lib."):
                candidates.append(d)
    for p in candidates:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)  # build directory will be inserted first, then lib.* directories
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
        from services.holographic_memory.api.app_soa import app  # type: ignore
        return app
    except Exception:
        pass
    try:
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
