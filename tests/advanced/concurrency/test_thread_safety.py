from __future__ import annotations

import threading
import pytest


@pytest.mark.performance
def test_basic_thread_safety(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    payload = b"thread-safe?"
    doc_ids: list[str] = []

    def worker():
        out = orch.store_content(payload, {"filename": "a.txt", "content_type": "text/plain"})
        doc_ids.append(out["doc_id"])

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(doc_ids) == 50

