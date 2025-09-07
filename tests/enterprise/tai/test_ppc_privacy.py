from __future__ import annotations

import numpy as np

from services.aiucp.quantum_core.enhanced_privacy import (
    gaussian_pld,
    compose_pld,
    pld_to_eps_delta,
    PrivacyBudget,
)


def test_pld_and_composition_and_epsilon():
    p1 = gaussian_pld(sigma=2.0, n=4097)  # coarse grid for speed
    p2 = gaussian_pld(sigma=1.5, n=4097)
    pc = compose_pld([p1, p2])
    eps = pld_to_eps_delta(pc, delta=1e-5)
    assert eps >= 0.0 and np.isfinite(eps)


def test_privacy_budget_tracking():
    budget = PrivacyBudget(eps=2.0, delta=1e-5)
    assert budget.can_spend(0.5, 5e-6)
    budget.spend(0.5, 5e-6)
    assert not budget.can_spend(2.0, 1e-5)

