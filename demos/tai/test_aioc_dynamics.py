from __future__ import annotations

import numpy as np

from services.aiucp.quantum_core.advanced_lyapunov import contraction_rate_field, ContractionFieldResult


def test_contraction_field_linear_jacobian():
    # Linear system x' = A x ⇒ J(x) = A
    A = np.array([[-1.0, 0.0], [0.0, -0.2]])

    def J(_x: np.ndarray) -> np.ndarray:
        return A

    grid = np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5], [-0.2, 0.1]])
    res: ContractionFieldResult = contraction_rate_field(J, grid)
    assert res.min_rate >= 0.0
    assert 0.0 <= res.grid_ok_fraction <= 1.0

