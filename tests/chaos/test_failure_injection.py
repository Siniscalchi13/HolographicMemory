from __future__ import annotations

import pytest


@pytest.mark.chaos
@pytest.mark.asyncio
@pytest.mark.parametrize("target", [
    "_get_gpu_status", "_get_memory_usage", "_get_performance_metrics", "_get_pattern_count", "_get_operations_count",
])
async def test_chaos_errors_in_metrics_are_caught(tmp_path, monkeypatch: pytest.MonkeyPatch, target: str):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)

    def boom(*_a, **_k):
        raise RuntimeError("chaos injected")

    monkeypatch.setattr(orch, target, boom, raising=True)
    out = await orch.get_real_metrics()
    assert isinstance(out, dict)
    assert "error" in out
    assert "timestamp" in out

