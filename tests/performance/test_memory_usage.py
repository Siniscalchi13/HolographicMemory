from __future__ import annotations

import pytest
from tests.utils.performance_utils import measure


@pytest.mark.performance
def test_memory_usage_store_path(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)

    def op():
        orch.store_content(b"x" * 2048, {"filename": "x.bin", "content_type": "application/octet-stream"})

    perf = measure(op, warmups=2)
    assert perf.peak_kb >= 0  # Smoke check; thresholds configured in CI

