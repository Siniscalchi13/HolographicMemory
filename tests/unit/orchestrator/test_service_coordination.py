from __future__ import annotations

import pytest


@pytest.mark.unit
def test_router_and_telemetry_are_used(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    res = orch.store_content(b"this is a document", {"filename": "doc.txt", "content_type": "text/plain"})
    assert res["routing_decision"]["format"] in {"micro", "microK8", "v4"}
    # compression tracking increments original and stored bytes
    orig, stored, ratio = orch.telemetry.current_ratios()
    assert orig >= stored >= 0
    # Rebalance call returns new dimensions
    rb = orch.rebalance_layers()
    assert set(rb.keys()) >= {"old_dimensions", "new_dimensions"}

