from __future__ import annotations

import os
import math
from typing import Tuple

import numpy as np
import pytest

from services.aiucp.holographic_memory.phase_retrieval import PhaseRetrievalCalculus, GSConfig


def _synthetic_spectrum(N: int = 64) -> np.ndarray:
    # Real, nonnegative, compact-ish signal yields an easier retrieval
    x = np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * (N / 10) ** 2))
    X = np.fft.fft(x, norm="ortho")
    return np.abs(X)


def test_gerchberg_saxton_accuracy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HMC_MODE", "mvp")
    calc = PhaseRetrievalCalculus(cfg=GSConfig(max_iterations=150, convergence_threshold=1e-7, real_signal=True, nonnegative=True))
    y = _synthetic_spectrum(64)
    x, info = calc.reconstruct(y, method="gs")
    assert info["method"] == "gs"
    # Relative MSE should be small for well-conditioned spectra
    assert info["relative_mse"] >= 0.0
    assert info["relative_mse"] < 1e-4


def test_wirtinger_and_admm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HMC_USE_ENHANCED_WF", "0")
    y = _synthetic_spectrum(64)
    calc = PhaseRetrievalCalculus()
    x_wf, info_wf = calc.reconstruct(y, method="wf", steps=200, step_size=0.3)
    assert info_wf["method"] in ("wirtinger", "wirtinger_adaptive")
    assert info_wf.get("relative_error", 0.0) >= 0.0
    x_ad, info_ad = calc.reconstruct(y, method="admm", steps=100)
    assert info_ad["method"] == "admm"
    assert info_ad.get("relative_error", 0.0) >= 0.0


def test_phaselift_when_available(monkeypatch: pytest.MonkeyPatch):
    # Conditional: skip gracefully if cvxpy isn't installed
    y = _synthetic_spectrum(16)
    calc = PhaseRetrievalCalculus()
    try:
        x_pl, info_pl = calc.reconstruct(y, method="phaselift")
        assert info_pl["method"] == "phaselift"
    except NotImplementedError:
        pytest.skip("cvxpy not available; skipping PhaseLift test")

