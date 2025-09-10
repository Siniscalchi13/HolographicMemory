from __future__ import annotations

import threading
import time
import pytest


@pytest.mark.performance
def test_no_deadlock_in_parallel_store(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    stop = False
    exc: list[Exception] = []

    def worker():
        while not stop:
            try:
                orch.store_content(b"x" * 32, {"filename": "x", "content_type": "application/octet-stream"})
            except Exception as e:
                exc.append(e)
                break

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    stop = True
    for t in threads:
        t.join(timeout=1.0)
    assert not exc

