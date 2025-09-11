"""
Dimension Optimization Service

Implements the closed-form allocation from Theorem 1.1 in
documentation/mathematical_foundations/HOLOGRAPHIC_7LAYER_THEORY.md
and the pseudocode in documentation/implementation/FORMULA_REFERENCE.md.
"""
from __future__ import annotations

from typing import Dict, Optional


class DimensionOptimizer:
    """
    Mathematical optimizer for layer dimensions D_k under total budget M.

    Given per-layer loads N_k and importance weights α_k, compute
    D_k* = M * (α_k^2 / N_k) / Σ_j (α_j^2 / N_j)
    with optional floors D_k ≥ S_k^2 N_k.
    """

    def optimize_dimensions(
        self,
        loads: Dict[str, int],
        importance: Dict[str, float],
        total_budget: int,
        floors: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        if total_budget <= 0:
            return {k: 0 for k in loads}
        # Compute q_k = α_k^2 / N_k (with guards for zeros)
        q: Dict[str, float] = {}
        for k, n in loads.items():
            n_eff = max(1, int(n))
            a = float(importance.get(k, 0.0))
            q[k] = (a * a) / float(n_eff)
        Z = sum(q.values()) or 1.0
        # Initial proportional allocation
        D = {k: int(round(total_budget * (q[k] / Z))) for k in q}
        # Apply floors (if any)
        if floors:
            # Apply floors to all keys that exist in both D and floors
            lifted = {}
            for k in q:
                floor_value = int(floors.get(k, 0))
                lifted[k] = max(D.get(k, 0), floor_value)
            lift_sum = sum(lifted.values())
            if lift_sum == total_budget:
                return lifted
            if lift_sum > total_budget:
                # Infeasible; scale down proportionally but keep ≥1 if originally positive
                scale = float(total_budget) / float(lift_sum)
                out: Dict[str, int] = {}
                for k, v in lifted.items():
                    new_v = int(max(0, round(v * scale)))
                    out[k] = new_v
                # Ensure sum equals budget by adjusting the largest entries if needed
                self._normalize_to_budget(out, total_budget)
                return out
            # Redistribute remaining budget according to q
            extra = total_budget - lift_sum
            add = {k: int(round(extra * (q[k] / (Z or 1.0)))) for k in q}
            out = {k: lifted.get(k, 0) + add.get(k, 0) for k in q}
            self._normalize_to_budget(out, total_budget)
            return out
        self._normalize_to_budget(D, total_budget)
        return D

    @staticmethod
    def _normalize_to_budget(D: Dict[str, int], budget: int) -> None:
        """Adjust values in-place so that Σ D_k == budget (small integer corrections)."""
        s = sum(D.values())
        if s == budget:
            return
        keys = list(D.keys())
        i = 0
        while s != budget and keys:
            k = keys[i % len(keys)]
            if s < budget:
                D[k] += 1
                s += 1
            else:
                if D[k] > 0:
                    D[k] -= 1
                    s -= 1
            i += 1

