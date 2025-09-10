from __future__ import annotations

import os
import time
import pytest


@pytest.mark.performance
def test_endurance_short_smoke(tmp_path):
    # Smoke endurance: loop for a few iterations; full endurance opt-in elsewhere
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    for i in range(100):
        orch.store_content(os.urandom(128), {"filename": f"s{i}", "content_type": "application/octet-stream"})
    assert True

