"""
# Mathematical Core Service

Purpose: Provide optimization and threshold calculations for the 7‑layer holographic memory per theory docs.

Key Docs:
- documentation/mathematical_foundations/HOLOGRAPHIC_7LAYER_THEORY.md
- documentation/implementation/FORMULA_REFERENCE.md

APIs (internal):
- DimensionOptimizer.optimize_dimensions(loads, importance, total_budget, floors?) -> dict[str,int]
- ThresholdCalculator.tau_star(c_micro, c_v4_curve, c_microk8=None, lo=1, hi=1<<20) -> int
- ThresholdCalculator.choose_format(size, c_micro, c_v4, c_microk8=None) -> str

Dependencies: NumPy (and optionally SciPy for future extensions).
"""

