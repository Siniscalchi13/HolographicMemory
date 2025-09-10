from __future__ import annotations

import numpy as np
from services.aiucp.quantum_core.analytics import compute_quantum_analytics


def _bell() -> np.ndarray:
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = rho[3, 3] = 0.5
    rho[0, 3] = rho[3, 0] = 0.5
    return rho


def test_qec_metrics_on_bell():
    rho = _bell()
    H = np.diag([1.0, 0.0, 0.0, -1.0])
    out = compute_quantum_analytics(rho, dims=(2, 2), H=H)
    assert abs(out.get("entropy_bits", 0.0)) < 1e-6
    assert 1.9 <= out.get("mutual_info_bits", 0.0) <= 2.1
    assert 0.45 <= out.get("negativity", 0.0) <= 0.55
    assert 0.95 <= out.get("concurrence", 0.0) <= 1.0
    assert 0.95 <= out.get("tangle", 0.0) <= 1.0
    assert abs(out.get("fidelity_self", 1.0) - 1.0) < 1e-12
    assert out.get("qfi_unitary", 0.0) >= 0.0

