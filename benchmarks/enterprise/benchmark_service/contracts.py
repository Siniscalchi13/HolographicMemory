from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Data contracts (SOA boundaries)
# -----------------------------


@dataclass(frozen=True)
class DataSpec:
    name: str
    num_items: int
    vector_dim: int
    field_shape: Tuple[int, int, int] = (32, 32, 32)  # 3D complex field
    seed: int = 1234


@dataclass(frozen=True)
class TextItem:
    id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    spec: DataSpec
    texts: List[TextItem]
    # Complex state vectors (n,) for each id
    states: Dict[str, "np.ndarray"]
    # 3D complex fields (Z,Y,X) for each id
    fields3d: Dict[str, "np.ndarray"]


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    category: str
    name: str
    metrics: List[Metric]
    notes: Optional[str] = None


@dataclass(frozen=True)
class PipelineResult:
    pipeline_name: str
    success_rate: float
    latency_ms_p50: float
    latency_ms_p95: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunReport:
    run_id: str
    platform: str
    dataset: DataSpec
    results: List[BenchmarkResult]
    pipelines: List[PipelineResult] = field(default_factory=list)


# type-checking-only import to avoid hard dep on NumPy at import time
try:  # pragma: no cover - dev convenience
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    pass

