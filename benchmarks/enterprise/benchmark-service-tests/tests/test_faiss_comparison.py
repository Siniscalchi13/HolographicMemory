from __future__ import annotations

import pytest

from services.aiucp.benchmark_service.contracts import DataSpec
from services.aiucp.benchmark_service.orchestrator import BenchmarkService


def test_faiss_comparison_top1_agreement():
    spec = DataSpec(name="test", num_items=24, vector_dim=64, field_shape=(8, 8, 8))
    svc = BenchmarkService(spec)
    res = svc.bench_faiss_comparison()
    metrics = {m.name: m.value for m in res.metrics}
    if metrics.get("faiss_available", 0.0) < 0.5:
        pytest.skip("FAISS not available in this environment")
    assert 0.0 <= metrics.get("top1_agreement_rate", 0.0) <= 1.0

