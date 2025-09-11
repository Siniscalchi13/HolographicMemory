from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.performance
@pytest.mark.parametrize("n,repeat", [(64, 10), (256, 10), (1024, 5)])
def test_store_latency_benchmark_light(benchmark, tmp_path: Path, n: int, repeat: int):  # type: ignore
    # light benchmark using orchestrator fake memory (patched by conftest)
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    data = os.urandom(n)

    def _run():
        for _ in range(repeat):
            orch.store_content(data, {"filename": "a.bin", "content_type": "application/octet-stream"})

    benchmark(_run)

