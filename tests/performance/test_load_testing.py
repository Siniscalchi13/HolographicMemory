from __future__ import annotations

import os
import random
import pytest


@pytest.mark.performance
def test_store_latency_benchmark(benchmark, tmp_path):  # type: ignore
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    payload = os.urandom(512)

    def do_store():
        orch.store_content(payload, {"filename": "blob.bin", "content_type": "application/octet-stream"})

    res = benchmark(do_store)
    # No hard assertion — report-only by default

