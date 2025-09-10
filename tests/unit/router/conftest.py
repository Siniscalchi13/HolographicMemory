from __future__ import annotations

import os
import pytest


@pytest.fixture(autouse=True)
def _router_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOLO_MICRO_THRESHOLD", "256")
    monkeypatch.setenv("HOLO_MICRO_K8_MAX", "1024")
    yield

