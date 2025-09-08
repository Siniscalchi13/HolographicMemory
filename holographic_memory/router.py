"""
Public SemanticRouter API

Re-exports the mathematically grounded router selecting layers and formats.
"""
from __future__ import annotations

try:
    from services.router import MathematicalRouter as _Router  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("services.router not available in distribution") from exc


class SemanticRouter(_Router):
    pass
