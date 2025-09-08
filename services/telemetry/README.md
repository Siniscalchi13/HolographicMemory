"""
# Telemetry Service

Purpose: Track compression metrics and per-layer loads to inform math_core optimizations.

APIs (internal):
- PerformanceTelemetry.track_compression(original: int, stored: int, layer: str) -> None
- PerformanceTelemetry.suggest_rebalancing(importance: dict[str,float], total_budget: int,
  floors: dict[str,int] | None = None) -> dict[str,int]
"""

