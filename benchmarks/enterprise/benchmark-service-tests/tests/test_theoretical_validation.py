from __future__ import annotations

import math

from services.aiucp.benchmark_service.services.theory_service import TheoryService


def test_theoretical_bounds():
    th = TheoryService(n=128)
    preds = th.predictions()
    # Tsirelson bound check
    assert math.isclose(preds.chsh_tsirelson, 2.0 * math.sqrt(2.0), rel_tol=1e-12)
    # Reasonable tolerance for FFT unitarity
    assert preds.fft_unitarity_err_tol <= 1e-8

