from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


def _orch(tmp_path: Path):
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    return HolographicMemoryOrchestrator(state_dir=tmp_path / "state", grid_size=64, use_gpu=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "size,filename,ctype",
    [
        (1, "a.txt", "text/plain"),
        (100, "b.md", "text/markdown"),
        (256, "c.cfg", "text/plain"),
        (600, "d.json", "application/json"),
        (1024, "e.pdf", "application/pdf"),
        (5000, "f.bin", "application/octet-stream"),
    ],
)
def test_store_content_formats(tmp_path: Path, size: int, filename: str, ctype: str) -> None:
    os.environ["HOLO_MICRO_THRESHOLD"] = "256"
    os.environ["HOLO_MICRO_K8_MAX"] = "1024"
    orch = _orch(tmp_path)
    payload = b"x" * size
    out = orch.store_content(payload, {"filename": filename, "content_type": ctype})
    assert out["doc_id"]
    assert out["routing_decision"]["format"] in {"micro", "microK8", "v4"}
    assert isinstance(out["layer_dimensions"], dict)


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        b"API_KEY=xyz", b"password=secret", b"Bearer abc.def.ghi", b"ghp_" + b"a" * 30,
    ],
)
def test_store_content_vault(tmp_path: Path, payload: bytes) -> None:
    orch = _orch(tmp_path)
    out = orch.store_content(payload, {"filename": "secrets.env", "content_type": "text/plain"})
    # When routed to vault, the structure is different
    assert out["encrypted"] is True
    assert out["holographic_patterns"] is False
    assert "vault_id" in out
    assert "vault_path" in out


@pytest.mark.unit
def test_retrieve_roundtrip(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    p = b"hello holographic"
    store = orch.store_content(p, {"filename": "a.txt", "content_type": "text/plain"})
    got = orch.retrieve_content(store["doc_id"])
    assert got["content"] == p


@pytest.mark.unit
def test_system_status_shape(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    status = orch.get_system_status()
    assert set(status.keys()) >= {"memory_status", "layer_dimensions", "current_loads", "telemetry"}


@pytest.mark.unit
def test_rebalance_changes_or_confirms(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    before = dict(orch.layer_dimensions)
    after = orch.rebalance_layers()
    assert set(after.keys()) == {"old_dimensions", "new_dimensions", "rebalancing_reason"}
    # Either changes or remains equal if optimizer yields same config
    assert isinstance(after["new_dimensions"], dict)


@pytest.mark.unit
@pytest.mark.parametrize("sz", [1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 100, 128, 200, 256, 300, 400, 512, 800, 1024, 1500])
def test_bulk_store_varied_sizes(tmp_path: Path, sz: int) -> None:
    orch = _orch(tmp_path)
    data = b"x" * sz
    out = orch.store_content(data, {"filename": f"f_{sz}.bin", "content_type": "application/octet-stream"})
    got = orch.retrieve_content(out["doc_id"]) 
    assert got["content"] == data


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_real_metrics", "get_system_status", "get_memory_usage", "get_performance_metrics"])
async def test_async_metrics_and_status_paths(tmp_path: Path, method: str) -> None:
    orch = _orch(tmp_path)
    fn = getattr(orch, method)
    out = await fn()
    assert isinstance(out, dict)
    assert "timestamp" in out or "available" in out or "ops_per_second" in str(out)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("target", [
    "_get_gpu_status", "_get_memory_usage", "_get_performance_metrics", "_get_pattern_count", "_get_operations_count",
])
async def test_async_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    orch = _orch(tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(orch, target, boom, raising=True)
    out = await orch.get_real_metrics()
    # On error path, still returns dict with timestamp and error key
    assert isinstance(out, dict)
    assert "timestamp" in out
    assert "error" in out
