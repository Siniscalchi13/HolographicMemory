from __future__ import annotations

import os
import pytest


@pytest.mark.performance
def test_large_payload_smoke(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)
    large = os.urandom(2 * 1024 * 1024)  # 2MB smoke
    out = orch.store_content(large, {"filename": "big.bin", "content_type": "application/octet-stream"})
    assert out["doc_id"]

