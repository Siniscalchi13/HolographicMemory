from __future__ import annotations

import pytest


class Boom(Exception):
    pass


@pytest.mark.unit
def test_vault_storage_path_on_secret(monkeypatch, tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    payload = b"api_key=XYZ; password=secret"
    out = orch.store_content(payload, {"filename": "secrets.txt", "content_type": "text/plain"})
    # When routed to vault, the structure is different
    assert out["encrypted"] is True
    assert out["holographic_patterns"] is False
    assert "vault_id" in out
    assert "vault_path" in out


@pytest.mark.unit
def test_retrieve_unknown_doc_returns_empty(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    got = orch.retrieve_content("deadbeefdeadbeef")
    assert got["content"] == b""

