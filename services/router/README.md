"""
# Router Service (Mathematical)

Purpose: Route content to layers using mathematical thresholds and security guardrails.

Depends on:
- services/math_core (DimensionOptimizer, ThresholdCalculator)
- services/vault (SecurityGuard)

Primary API:
- MathematicalRouter.route_content(content: bytes, metadata: dict) -> dict
  Returns routing decision: format (micro/v4/microK8), layers, vault flag, K.
"""

