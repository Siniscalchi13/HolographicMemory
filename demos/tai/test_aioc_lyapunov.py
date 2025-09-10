from __future__ import annotations

import numpy as np

from services.aiucp.quantum_core.advanced_lyapunov import quadratic_stability_region, passivity_kyp


def test_quadratic_stability_region_linear_system():
    # Stable 2x2 system (Hurwitz)
    A = np.array([[-1.0, 0.2], [-0.1, -0.5]])
    reg = quadratic_stability_region(A, samples=100)
    # Some level sets should be certified
    assert reg.ok_mask.any()
    # P must be symmetric positive definite numerically
    w = np.linalg.eigvalsh((reg.P + reg.P.T) / 2.0)
    assert (w > 0).all()


def test_passivity_kyp_fallback():
    # Simple passive system candidate
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.5]])
    out = passivity_kyp(A, B, C, D)
    assert "ok" in out and "min_eig_symD" in out

