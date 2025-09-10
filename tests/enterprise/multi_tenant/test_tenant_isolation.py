from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.integration
def test_tenant_isolation_via_separate_state_dirs(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    t1 = HolographicMemoryOrchestrator(state_dir=tmp_path / "t1", grid_size=64, use_gpu=False)
    t2 = HolographicMemoryOrchestrator(state_dir=tmp_path / "t2", grid_size=64, use_gpu=False)
    a = t1.store_content(b"alpha", {"filename": "a", "content_type": "text/plain"})["doc_id"]
    b = t2.store_content(b"beta", {"filename": "b", "content_type": "text/plain"})["doc_id"]
    assert t1.retrieve_content(b)["content"] == b""  # no cross-tenant leaks

