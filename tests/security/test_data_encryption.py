from __future__ import annotations

import pytest


@pytest.mark.security
def test_vault_storage_marks_encrypted(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    res = orch.store_content(b"token=abc", {"filename": "auth.env", "content_type": "text/plain"})
    assert res["encrypted"] is True
    assert res["holographic_patterns"] is False
    assert "vault_id" in res
    assert "vault_path" in res

