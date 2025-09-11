"""
Threshold and Format Decision Service

Implements τ* crossover selection per documentation/implementation/FORMULA_REFERENCE.md.
"""
from __future__ import annotations

from typing import Callable, Optional


class ThresholdCalculator:
    """Compute crossover thresholds and choose storage formats.

    Glossary:
      - c_micro: constant cost (bytes) for micro header format
      - c_microk8: optional constant cost for micro+coeffs (K=8 packed)
      - c_v4_curve: monotone cost function c_v4(size) for sparse v4 path
    """

    def tau_star(
        self,
        c_micro: int,
        c_v4_curve: Callable[[int], int],
        c_microk8: Optional[int] = None,
        lo: int = 1,
        hi: int = 1 << 20,
    ) -> int:
        """Find τ* = min{s: c_v4(s) ≤ min(c_micro, c_microk8)} via binary search.

        Assumes c_v4 is non‑increasing or slowly varying with size (piecewise constant
        header + K·bytes dominates), so a simple binary search suffices within
        practical ranges.
        """
        target = min(int(c_micro), int(c_microk8)) if c_microk8 is not None else int(c_micro)
        lo = max(1, int(lo))
        hi = max(lo, int(hi))
        
        # Check if even the highest value meets the target
        if int(c_v4_curve(hi)) > target:
            return hi  # Return hi if no value meets the target
        
        ans = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            cv = int(c_v4_curve(mid))
            if cv <= target:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return int(ans)

    @staticmethod
    def choose_format(
        size: int,
        c_micro: int,
        c_v4: int,
        c_microk8: Optional[int] = None,
    ) -> str:
        """Choose minimal‑cost format among micro, v4, and optional microK8.

        Returns one of: "micro", "v4", or "microK8".
        """
        best_name = "micro"
        best_cost = int(c_micro)
        if c_microk8 is not None and int(c_microk8) < best_cost:
            best_name, best_cost = "microK8", int(c_microk8)
        if int(c_v4) < best_cost:
            best_name, best_cost = "v4", int(c_v4)
        return best_name

