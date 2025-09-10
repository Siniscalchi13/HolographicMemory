from __future__ import annotations

import os
import pytest


@pytest.mark.performance
def test_spike_load_smoke(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    for _ in range(500):
        orch.store_content(os.urandom(64), {"filename": "spike", "content_type": "application/octet-stream"})
    assert True

