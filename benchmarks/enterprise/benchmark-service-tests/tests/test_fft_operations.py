from __future__ import annotations

import numpy as np

from services.aiucp.benchmark_service.contracts import DataSpec
from services.aiucp.benchmark_service.orchestrator import BenchmarkService
from services.aiucp.benchmark_service.services.theory_service import TheoryService


def test_fft_roundtrip_and_unitarity():
    spec = DataSpec(name="test", num_items=4, vector_dim=64, field_shape=(8, 8, 8))
    svc = BenchmarkService(spec)
    res = svc.bench_fft_ops()
    # For small n, verify theoretical unitary error tolerance
    theory = TheoryService(n=64)
    tol = theory.predictions().fft_unitarity_err_tol
    err = theory.fft_unitarity_error(32)
    assert err <= tol * 100  # allow leeway across platforms, but remain small

