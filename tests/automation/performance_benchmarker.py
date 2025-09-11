from __future__ import annotations

import os
import time
from pathlib import Path


def run_store_benchmark(state_dir: Path, size: int, repeat: int = 10) -> float:
    from services.orchestrator.orchestrator import HolographicMemoryOrchestrator

    orch = HolographicMemoryOrchestrator(state_dir=state_dir, grid_size=64, use_gpu=False)
    data = os.urandom(size)
    t0 = time.perf_counter()
    for _ in range(repeat):
        orch.store_content(data, {"filename": "a.bin", "content_type": "application/octet-stream"})
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, repeat)


if __name__ == "__main__":
    out = run_store_benchmark(Path(".tmp/bench"), 256, 10)
    print({"avg_seconds_per_store": out})

