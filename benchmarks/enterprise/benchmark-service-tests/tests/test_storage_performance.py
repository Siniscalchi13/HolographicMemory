from __future__ import annotations

from services.aiucp.benchmark_service.contracts import DataSpec
from services.aiucp.benchmark_service.orchestrator import BenchmarkService


def test_storage_performance_small():
    spec = DataSpec(name="test", num_items=16, vector_dim=128, field_shape=(16, 16, 16))
    svc = BenchmarkService(spec)
    res = svc.bench_storage()
    metrics = {m.name: m.value for m in res.metrics}
    assert metrics["stored_count"] == spec.num_items
    assert metrics["throughput"] >= 1.0  # trivial lower bound

