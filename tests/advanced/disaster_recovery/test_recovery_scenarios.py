from __future__ import annotations

import pytest


@pytest.mark.integration
def test_vault_route_serves_as_secure_fallback(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    res = orch.store_content(b"token=abc", {"filename": "secrets.env", "content_type": "text/plain"})
    assert res["routing_decision"]["vault"] is True
    assert res["storage_result"]["encrypted"] is True

