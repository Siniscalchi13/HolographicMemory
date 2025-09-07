from __future__ import annotations

import time
import numpy as np

from services.aiucp.verbum_field_engine.selection import selection_scores
from services.aiucp.holographic_memory.pure_python_memory import HolographicMemory
from services.aiucp.quantum_core.analytics import compute_quantum_analytics
from services.aiucp.quantum_core.advanced_lyapunov import quadratic_stability_region


def test_cross_service_latency_sanity():
    # VFE selection
    models = [
        {"name": "apple-small", "backend": "apple", "ctx": 4096, "A": [[0.2, 0.3, 0.1, 0.0, 0.5]]},
        {"name": "llama-fast", "backend": "llama.cpp", "ctx": 8192, "A": [[0.1, 0.6, 0.2, 0.0, 0.3]]},
    ]
    msgs = [{"role": "user", "content": "Short prompt for selection."}]
    t0 = time.perf_counter()
    sel, scores = selection_scores(models, msgs, max_tokens=64)
    dt_sel = time.perf_counter() - t0
    assert sel in scores and dt_sel < 0.2

    # HMC memory operations
    mem = HolographicMemory()
    t1 = time.perf_counter()
    doc_id = mem.store(b"short text")
    _ = mem.retrieve(doc_id)
    res = mem.search_semantic("text", top_k=1)
    dt_mem = time.perf_counter() - t1
    assert res and dt_mem < 0.1

    # QEC analytics
    rho = np.eye(4) / 4.0
    t2 = time.perf_counter()
    out = compute_quantum_analytics(rho, dims=(2, 2))
    dt_qec = time.perf_counter() - t2
    assert "entropy_bits" in out and dt_qec < 0.05

    # Lyapunov region
    A = np.array([[-1.0, 0.2], [-0.1, -0.5]])
    t3 = time.perf_counter()
    reg = quadratic_stability_region(A, samples=50)
    dt_aioc = time.perf_counter() - t3
    assert reg.ok_mask.any() and dt_aioc < 0.4

