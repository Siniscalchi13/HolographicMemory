from __future__ import annotations

from services.aiucp.benchmark_service.contracts import DataSpec
from services.aiucp.benchmark_service.orchestrator import BenchmarkService


def test_full_pipeline_embeddings_storage_retrieval():
    spec = DataSpec(name="test", num_items=20, vector_dim=128, field_shape=(8, 8, 8))
    svc = BenchmarkService(spec)
    pipe = svc.pipeline_embeddings_storage_retrieval()
    assert pipe.success_rate >= 0.8
    assert pipe.latency_ms_p50 >= 0.0

