from __future__ import annotations

import pytest


@pytest.mark.security
def test_vault_storage_marks_encrypted(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    res = orch.store_content(b"token=abc", {"filename": "auth.env", "content_type": "text/plain"})
    assert res["routing_decision"]["vault"] is True
    assert res["storage_result"]["encrypted"] is True

