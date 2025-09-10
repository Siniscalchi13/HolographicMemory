from __future__ import annotations

import os
import pytest


@pytest.mark.performance
def test_stress_store_spike_opt_in(tmp_path):
    if os.getenv("ENABLE_HEAVY_PERF") != "1":
        pytest.skip("ENABLE_HEAVY_PERF not set; skipping stress test")
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    for i in range(2000):
        orch.store_content(os.urandom(1024), {"filename": f"f{i}.bin", "content_type": "application/octet-stream"})
    assert True

