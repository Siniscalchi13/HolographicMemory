from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_store_retrieve_data_integrity(tmp_path: Path) -> None:
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state")
    blob = b"e2e-data-integrity-check"
    out = orch.store_content(blob, {"filename": "d.bin", "content_type": "application/octet-stream"})
    doc_id = out["doc_id"]
    got = orch.retrieve_content(doc_id)
    assert got["content"] == blob

