from __future__ import annotations

from services.aiucp.benchmark_service.contracts import DataSpec
from services.aiucp.benchmark_service.orchestrator import BenchmarkService


def test_retrieval_accuracy_top1():
    spec = DataSpec(name="test", num_items=24, vector_dim=128, field_shape=(16, 16, 16))
    svc = BenchmarkService(spec)
    res = svc.bench_retrieval_accuracy()
    metrics = {m.name: m.value for m in res.metrics}
    # With deterministic state encoding + exact queries we expect high accuracy
    assert metrics["top1_success_rate"] >= 0.85

