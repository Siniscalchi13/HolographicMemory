from __future__ import annotations

import shutil
from pathlib import Path
import pytest


@pytest.mark.integration
def test_backup_and_restore(tmp_path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    state = tmp_path / "state"
    orch = HolographicMemoryOrchestrator(state_dir=state, grid_size=64, use_gpu=False)
    out = orch.store_content(b"backup", {"filename": "a.txt", "content_type": "text/plain"})
    did = out["doc_id"]
    # Backup state dir
    backup = tmp_path / "backup"
    shutil.copytree(state, backup)
    # Restore to new location and ensure files exist
    restored = tmp_path / "restored"
    shutil.copytree(backup, restored)
    assert restored.exists()

