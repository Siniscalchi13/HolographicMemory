from __future__ import annotations

import threading
import pytest


@pytest.mark.performance
def test_no_race_on_retrieve_missing(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    errors: list[Exception] = []

    def worker():
        try:
            orch.retrieve_content("missing")
        except Exception as e:  # should not raise
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

