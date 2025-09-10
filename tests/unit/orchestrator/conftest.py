from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import pytest


@dataclass
class _FakeMemory:
    state: Dict[str, bytes] = field(default_factory=dict)
    grid_size: int = 64
    use_gpu: bool = False
    backend = object()  # sentinel

    def __init__(self, state_dir, grid_size: int = 64, use_gpu: bool = False) -> None:  # noqa: D401
        self.state = {}
        self.grid_size = int(grid_size)
        self.use_gpu = bool(use_gpu)

    def store_bytes(self, doc_id: str, content: bytes) -> Dict[str, Any]:
        self.state[str(doc_id)] = bytes(content)
        return {"ok": True, "encoded_data": content[:16]}

    def retrieve_bytes(self, doc_id: str) -> bytes:
        return self.state.get(str(doc_id), b"")


@pytest.fixture(autouse=True)
def patch_holographic_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch orchestrator to use a fake memory backend to avoid native deps."""
    try:
        import services.orchestrator.orchestrator as orch
    except Exception:
        import services.orchestrator.orchestrator as orch  # type: ignore[no-redef]
    monkeypatch.setattr(orch, "HolographicMemory", _FakeMemory, raising=True)

