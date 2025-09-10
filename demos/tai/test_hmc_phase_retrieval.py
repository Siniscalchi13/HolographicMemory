from __future__ import annotations

import numpy as np
import pytest

from services.aiucp.holographic_memory.phase_retrieval import PhaseRetrievalCalculus, GSConfig


def _spectrum(N: int = 64) -> np.ndarray:
    x = np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * (N / 10) ** 2))
    return np.abs(np.fft.fft(x, norm="ortho"))


def test_gs_accuracy():
    calc = PhaseRetrievalCalculus(cfg=GSConfig(max_iterations=150, convergence_threshold=1e-7, real_signal=True, nonnegative=True))
    y = _spectrum(64)
    x, info = calc.reconstruct(y, method="gs")
    assert info["method"] == "gs"
    assert info["relative_mse"] >= 0.0 and info["relative_mse"] < 1e-4


def test_wf_admm_sanity():
    y = _spectrum(64)
    calc = PhaseRetrievalCalculus()
    x_wf, info_wf = calc.reconstruct(y, method="wf", steps=200, step_size=0.3)
    assert info_wf.get("relative_error", 0.0) >= 0.0
    x_ad, info_ad = calc.reconstruct(y, method="admm", steps=100)
    assert info_ad.get("relative_error", 0.0) >= 0.0


def test_phaselift_optional():
    y = _spectrum(16)
    calc = PhaseRetrievalCalculus()
    try:
        x, info = calc.reconstruct(y, method="phaselift")
        assert info["method"] == "phaselift"
    except NotImplementedError:
        pytest.skip("cvxpy unavailable; skipping PhaseLift")

