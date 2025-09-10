from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_store_and_retrieve_roundtrip(tmp_path: Path) -> None:
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    payload = b"hello-hmc"
    out = orch.store_content(payload, {"filename": "greeting.txt", "content_type": "text/plain"})
    doc_id = out["doc_id"]
    assert isinstance(doc_id, str) and len(doc_id) == 16
    got = orch.retrieve_content(doc_id)
    assert got["content"] == payload


@pytest.mark.unit
def test_get_system_status_contains_expected_keys(tmp_path: Path) -> None:
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    st = orch.get_system_status()
    assert set(st.keys()) >= {"memory_status", "layer_dimensions", "current_loads", "telemetry"}
    assert "grid_size" in st["memory_status"]

