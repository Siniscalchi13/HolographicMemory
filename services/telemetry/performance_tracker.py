"""
Performance Telemetry

Collects compression stats and layer loads; produces rebalancing suggestions
using DimensionOptimizer.
"""
from __future__ import annotations

from typing import Dict, Tuple
import sys
from pathlib import Path

# Add math-core to path
_services_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_services_root / "math-core"))

from optimizer import DimensionOptimizer


class PerformanceTelemetry:
    def __init__(self) -> None:
        self._loads: Dict[str, int] = {}
        self._bytes_original: int = 0
        self._bytes_stored: int = 0
        self._bytes_by_layer_orig: Dict[str, int] = {}
        self._bytes_by_layer_stored: Dict[str, int] = {}
        self._retrievals: int = 0
        self._optimizer = DimensionOptimizer()

    def track_compression(self, original: int, stored: int, layer: str) -> None:
        self._bytes_original += int(max(0, original))
        self._bytes_stored += int(max(0, stored))
        self._loads[layer] = self._loads.get(layer, 0) + 1
        self._bytes_by_layer_orig[layer] = self._bytes_by_layer_orig.get(layer, 0) + int(max(0, original))
        self._bytes_by_layer_stored[layer] = self._bytes_by_layer_stored.get(layer, 0) + int(max(0, stored))

    def track_retrieval(self) -> None:
        self._retrievals += 1

    def current_ratios(self) -> Tuple[int, int, float | None]:
        if self._bytes_stored <= 0:
            return (self._bytes_original, self._bytes_stored, None)
        ratio = float(self._bytes_original) / float(self._bytes_stored)
        return (self._bytes_original, self._bytes_stored, ratio)

    def suggest_rebalancing(
        self,
        importance: Dict[str, float],
        total_budget: int,
        floors: Dict[str, int] | None = None,
    ) -> Dict[str, int]:
        return self._optimizer.optimize_dimensions(self._loads, importance, int(total_budget), floors)

    def snapshot(self) -> Dict:
        """Return a snapshot of telemetry metrics suitable for APIs."""
        overall_orig, overall_stored, ratio = self.current_ratios()
        per_layer: Dict[str, Dict[str, float | int | None]] = {}
        total_count = sum(self._loads.values()) or 1
        for layer, cnt in self._loads.items():
            s = int(self._bytes_by_layer_stored.get(layer, 0))
            o = int(self._bytes_by_layer_orig.get(layer, 0))
            r = (float(o) / float(s)) if s > 0 else None
            per_layer[layer] = {
                "count": int(cnt),
                "utilization_pct": (100.0 * float(cnt) / float(total_count)),
                "bytes_original": o,
                "bytes_stored": s,
                "compression_x": r,
            }
        return {
            "overall": {
                "bytes_original": overall_orig,
                "bytes_stored": overall_stored,
                "compression_x": ratio,
                "retrievals": self._retrievals,
            },
            "per_layer": per_layer,
            "loads": dict(self._loads),
        }
