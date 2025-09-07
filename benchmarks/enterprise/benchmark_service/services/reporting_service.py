from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..contracts import BenchmarkResult, PipelineResult, RunReport


class ReportingService:
    """
    Writes JSON and (optionally) HDF5 results to reports/benchmarks.
    """

    def __init__(self, reports_dir: str | Path = "reports/benchmarks") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _ts(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def write_json(self, report: RunReport) -> Path:
        path = self.reports_dir / f"benchmark_service_{report.run_id}.json"
        serial = asdict(report)
        # Convert dataclass Metrics to plain dicts (asdict handled nested dataclasses)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2)
        return path

    def write_hdf5(self, arrays: Dict[str, np.ndarray], run_id: str) -> Path | None:
        try:
            import h5py  # type: ignore
        except Exception:
            return None
        path = self.reports_dir / f"benchmark_service_{run_id}.h5"
        with h5py.File(path, "w") as h5:
            for name, arr in arrays.items():
                h5.create_dataset(name, data=arr)
        return path

