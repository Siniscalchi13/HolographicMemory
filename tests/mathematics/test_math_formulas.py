"""
Mathematical validation tests (backend-independent).

Validates core formulas used by the holographic memory design without
requiring the C++ CPU/GPU backends to be built or importable.

Covered:
- Theorem 1.1 (optimal dimension allocation)
- SNR engineering rule (SNR_k ≈ sqrt(D_k / N_k))
- Capacity theorem bound (D_k ≥ S_k² N_k)
- CHSH Bell inequality (|S| ≤ 2√2, violation > 2 at quantum angles)
"""

from __future__ import annotations

import math
from typing import List


def optimal_allocation(alphas: List[float], loads: List[int], M: int) -> List[int]:
    """Closed-form allocation D_k* = M * (α_k^2 / N_k) / Σ_j (α_j^2 / N_j).

    - Uses guards to avoid division by zero (treat N_k<=0 as 1).
    - Rounds to nearest int, then adjusts small rounding errors to sum to M.
    """
    assert len(alphas) == len(loads) and M >= 0
    n = len(alphas)
    if n == 0 or M == 0:
        return [0] * n
    q = []
    for a, Nk in zip(alphas, loads):
        n_eff = max(1, int(Nk))
        q.append((float(a) * float(a)) / float(n_eff))
    Z = sum(q) or 1.0
    D = [int(round(M * (qi / Z))) for qi in q]
    # Normalize to sum exactly M
    diff = M - sum(D)
    i = 0
    while diff != 0 and n > 0:
        j = i % n
        if diff > 0:
            D[j] += 1
            diff -= 1
        else:
            if D[j] > 0:
                D[j] -= 1
                diff += 1
        i += 1
    return D


def chsh_S(a: float, ap: float, b: float, bp: float) -> float:
    """CHSH S-value using singlet correlation E(θ) = -cos(θ)."""
    def E(x: float, y: float) -> float:
        return -math.cos(x - y)
    return E(a, b) + E(a, bp) + E(ap, b) - E(ap, bp)


def test_theorem_1_1_allocation_equal_loads():
    # α = [1,2,1], N = [100,100,100], M=600 → q ∝ [1,4,1] → D ≈ [100,400,100]
    alphas = [1.0, 2.0, 1.0]
    loads = [100, 100, 100]
    M = 600
    D = optimal_allocation(alphas, loads, M)
    assert sum(D) == M
    assert D == [100, 400, 100]


def test_theorem_1_1_allocation_varied_loads():
    # α = [1,1,1], N = [50,100,150], M=600
    # q ∝ [1/50, 1/100, 1/150] → dims roughly [327,164,109]
    alphas = [1.0, 1.0, 1.0]
    loads = [50, 100, 150]
    M = 600
    D = optimal_allocation(alphas, loads, M)
    assert sum(D) == M
    # sanity: more load should get fewer dimensions (ceteris paribus)
    assert D[0] > D[1] > D[2]


def test_snr_calculation_rule_of_thumb():
    # SNR_k ≈ sqrt(D_k / N_k)
    cases = [
        (200, 50, 2.0),     # sqrt(4) = 2
        (128, 512, 0.5),    # sqrt(0.25) = 0.5
        (1, 1, 1.0),
        (1000, 1000, 1.0),
    ]
    for D, N, expected in cases:
        snr = math.sqrt(max(0.0, float(D) / float(max(1, N))))
        assert abs(snr - expected) < 1e-9


def test_capacity_theorem_boundary():
    # Capacity floor: D_k ≥ S_k² N_k
    N = 50
    S = 2.0
    required = int(math.ceil(S * S * N))  # 200
    assert required == 200
    assert (199 >= required) is False
    assert (200 >= required) is True
    assert (250 >= required) is True


def test_bell_inequality_chsh():
    # Quantum-optimal settings: a=0, a'=π/2, b=π/4, b'=-π/4 → |S| = 2√2
    a, ap, b, bp = 0.0, math.pi / 2.0, math.pi / 4.0, -math.pi / 4.0
    S = abs(chsh_S(a, ap, b, bp))
    assert S > 2.0 - 1e-9
    assert S <= 2.0 * math.sqrt(2.0) + 1e-9

