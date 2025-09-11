from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pytest


@pytest.mark.performance
def test_concurrent_store_operations(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)

    def task(i: int) -> str:
        data = f"Test content for file {i} - this is low entropy text that won't trigger security guard".encode()
        out = orch.store_content(data, {"filename": f"f{i}.txt", "content_type": "text/plain"})
        return out["doc_id"]

    N = 200  # Keep moderate by default; scale under dedicated perf env
    ids = set()
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(task, i) for i in range(N)]
        for f in as_completed(futs):
            ids.add(f.result())
    assert len(ids) == N

