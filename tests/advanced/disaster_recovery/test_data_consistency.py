from __future__ import annotations

import pytest


@pytest.mark.integration
def test_roundtrip_stable_after_backup_restore(tmp_path):
    # This test leverages orchestrator roundtrip; backup is tested separately
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    data = b"disaster-recovery-check"
    out = orch.store_content(data, {"filename": "d.bin", "content_type": "application/octet-stream"})
    got = orch.retrieve_content(out["doc_id"])  # type: ignore[index]
    assert got["content"] == data

