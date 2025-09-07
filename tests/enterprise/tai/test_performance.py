from __future__ import annotations

import time
import numpy as np

from services.aiucp.verbum_field_engine.selection import selection_scores
from services.aiucp.quantum_core.analytics import compute_quantum_analytics
from services.aiucp.verbum_field_engine.nsga2_optimizer import nsga2, NSGA2Config
from services.aiucp.holographic_memory.phase_retrieval import PhaseRetrievalCalculus, GSConfig


def test_selection_latency(sample_models, messages_chat):
    t0 = time.perf_counter()
    sel, scores = selection_scores(sample_models, messages_chat, max_tokens=128)
    dt = time.perf_counter() - t0
    assert sel in scores
    assert dt < 0.25  # seconds


def test_qec_analytics_latency():
    rho = np.eye(4) / 4.0
    H = np.diag([1.0, 0.0, 0.0, -1.0])
    t0 = time.perf_counter()
    out = compute_quantum_analytics(rho, dims=(2, 2), H=H)
    dt = time.perf_counter() - t0
    assert "entropy_bits" in out and dt < 0.05


def _spectrum(N: int = 64) -> np.ndarray:
    x = np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * (N / 10) ** 2))
    return np.abs(np.fft.fft(x, norm="ortho"))


def test_phase_retrieval_latency():
    calc = PhaseRetrievalCalculus(cfg=GSConfig(max_iterations=120, real_signal=True, nonnegative=True))
    y = _spectrum(64)
    t0 = time.perf_counter()
    x, info = calc.reconstruct(y, method="gs")
    dt = time.perf_counter() - t0
    assert info["relative_mse"] >= 0.0 and dt < 0.5


def test_nsga2_latency():
    def f(x: np.ndarray) -> np.ndarray:
        t = x[0]
        return np.array([t * t, (t - 1.0) ** 2])

    low = np.array([0.0])
    high = np.array([1.0])
    cfg = NSGA2Config(pop_size=16, n_gen=10)
    t0 = time.perf_counter()
    front = nsga2(f, bounds=(low, high), n_var=1, cfg=cfg)
    dt = time.perf_counter() - t0
    assert len(front) >= 4 and dt < 1.0

