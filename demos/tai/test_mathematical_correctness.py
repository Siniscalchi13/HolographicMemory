from __future__ import annotations

import math
import numpy as np

from services.aiucp.verbum_field_engine.pareto import pac_bayes_mcallester
from services.aiucp.verbum_field_engine.nsga2_optimizer import nsga2, NSGA2Config, benchmark_vs_chebyshev


def test_pac_bayes_bound_monotonicity():
    # As empirical risk decreases or m increases, bound should not increase
    b1 = pac_bayes_mcallester(emp_risk=0.3, kl=0.1, m=100, delta=0.05)
    b2 = pac_bayes_mcallester(emp_risk=0.2, kl=0.1, m=100, delta=0.05)
    b3 = pac_bayes_mcallester(emp_risk=0.3, kl=0.1, m=500, delta=0.05)
    assert 0.0 <= b1 <= 1.0
    assert b2 <= b1 + 1e-12
    assert b3 <= b1 + 1e-12


def test_nsga2_front_properties():
    # 1-D decision variable, 2 objectives (convex trade-off)
    def f(x: np.ndarray) -> np.ndarray:
        t = x[0]
        return np.array([t * t, (t - 1.0) ** 2])

    low = np.array([0.0])
    high = np.array([1.0])
    front = nsga2(f, bounds=(low, high), n_var=1, cfg=NSGA2Config(pop_size=32, n_gen=20))
    # Non-dominated rank-0 front should cover the curve
    F = np.vstack([ind.f for ind in front])
    assert F.shape[0] >= 5
    # Check spread positive across both objectives
    assert float(np.ptp(F[:, 0])) > 0.0
    assert float(np.ptp(F[:, 1])) > 0.0


def test_nsga2_chebyshev_benchmark():
    def f(x: np.ndarray) -> np.ndarray:
        t = x[0]
        return np.array([t * t, (t - 1.0) ** 2])

    low = np.array([0.0])
    high = np.array([1.0])
    metrics = benchmark_vs_chebyshev(f, bounds=(low, high), n_var=1, m=2, n_points=32)
    assert metrics["spread"] > 0.0
    assert metrics["coverage"] >= 0.0

