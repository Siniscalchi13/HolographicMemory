from __future__ import annotations

import os
import pytest


@pytest.fixture(autouse=True)
def _disable_gpu_for_unit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOLOGRAPHIC_USE_GPU", "false")
    yield

