from __future__ import annotations

import pytest


@pytest.mark.chaos
def test_orchestrator_handles_missing_doc_gracefully(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    out = orch.retrieve_content("nonexistent")
    # Should return structure with empty content and a small retrieval_time
    assert out["content"] == b""
    assert out["retrieval_time"] >= 0.0

