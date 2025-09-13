from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUSample:
    available: bool
    utilization: float  # percent 0..100
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None
    provider: str = "none"


class GPUProvider:
    """Attempts to read GPU telemetry from available providers.

    Priority:
      1) holographic_gpu native module (if exposes metrics)
      2) Environment-driven simulation for demos (GPU_SIM_UTIL, GPU_SIM_MEM_GB)
      3) No GPU available -> zeros
    """

    def __init__(self) -> None:
        self._hg = None
        self._hg_gpu = None
        # Try holographic_gpu
        try:
            import holographic_gpu as _hg  # type: ignore

            self._hg = _hg
            # Create a GPU instance if possible
            if hasattr(_hg, "HolographicGPU"):
                try:
                    gpu = _hg.HolographicGPU()
                    # Some builds need initialize()
                    ok = True
                    if hasattr(gpu, "initialize"):
                        ok = bool(gpu.initialize())
                    self._hg_gpu = gpu if ok else None
                except Exception:
                    self._hg_gpu = None
        except Exception:
            self._hg = None
            self._hg_gpu = None

    def sample(self) -> GPUSample:
        # 1) Try holographic_gpu if instantiated
        if self._hg_gpu is not None:
            util: Optional[float] = None
            used: Optional[float] = None
            total: Optional[float] = None
            # Try common method names in priority order
            for name in (
                "get_utilization",
                "utilization",
                "get_gpu_utilization",
                "gpu_utilization",
            ):
                try:
                    fn = getattr(self._hg_gpu, name)
                except AttributeError:
                    fn = None
                if fn:
                    try:
                        v = fn()
                        util = float(v)
                        break
                    except Exception:
                        pass

            # Memory info attempts
            # Expect either a tuple (used,total) GB, or methods returning GB
            mem_attempts = (
                ("get_memory_info_gb", None),  # returns (used,total)
                ("get_memory_used_gb", "get_memory_total_gb"),
                ("memory_used_gb", "memory_total_gb"),  # attributes
            )
            for u_name, t_name in mem_attempts:
                try:
                    if t_name is None:
                        fn = getattr(self._hg_gpu, u_name)
                        tup = fn()
                        used, total = float(tup[0]), float(tup[1])
                        break
                    else:
                        u = getattr(self._hg_gpu, u_name)
                        t = getattr(self._hg_gpu, t_name)
                        used = float(u() if callable(u) else u)
                        total = float(t() if callable(t) else t)
                        break
                except Exception:
                    continue

            return GPUSample(
                available=True,
                utilization=max(0.0, min(100.0, util if util is not None else 0.0)),
                memory_used_gb=used,
                memory_total_gb=total,
                provider="holographic_gpu",
            )

        # 2) Environment-driven simulation (for demos/tests)
        try:
            sim_util = float(os.environ.get("GPU_SIM_UTIL", "nan"))
        except ValueError:
            sim_util = float("nan")
        try:
            sim_mem = float(os.environ.get("GPU_SIM_MEM_GB", "nan"))
        except ValueError:
            sim_mem = float("nan")

        if sim_util == sim_util or sim_mem == sim_mem:  # nan check
            return GPUSample(
                available=True,
                utilization=max(0.0, min(100.0, sim_util if sim_util == sim_util else 0.0)),
                memory_used_gb=(sim_mem if sim_mem == sim_mem else None),
                memory_total_gb=None,
                provider="sim_env",
            )

        # 3) No GPU available
        return GPUSample(available=False, utilization=0.0, provider="none")

