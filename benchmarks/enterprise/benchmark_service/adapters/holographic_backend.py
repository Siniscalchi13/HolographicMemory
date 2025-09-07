from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    # Prefer optimized native modules if present
    import holographic_native as _holo_native  # type: ignore
    HAVE_NATIVE = True
except Exception:
    HAVE_NATIVE = False

try:  # fallback pure-Python components from our service tree
    from services.aiucp.holographic_memory.pure_python_memory import HolographicMemory
    from services.aiucp.holographic_memory.wave_functions import HolographicResonanceIndex
except Exception:  # pragma: no cover
    # Very minimal fallbacks if import path not available
    HolographicMemory = None  # type: ignore
    HolographicResonanceIndex = None  # type: ignore


@dataclass
class StoreStats:
    count: int
    duration_ms: float
    throughput_items_per_s: float


class HoloBackend:
    """
    Adapter for the holographic backend.

    Provides:
    - Simple document store with exact recall (Python fallback)
    - Resonance index for vector search (Python fallback)
    - FFT ops on 1D/2D/3D arrays using NumPy (Accelerate on macOS)
    - Optional native C++ bindings if available
    """

    def __init__(self, vector_dim: int = 256, field_shape: Tuple[int, int, int] = (32, 32, 32)) -> None:
        self.vector_dim = int(vector_dim)
        self.field_shape = tuple(map(int, field_shape))
        self._mem = HolographicMemory() if HolographicMemory is not None else None
        self._res_index = (
            HolographicResonanceIndex(dimension=vector_dim) if HolographicResonanceIndex is not None else None
        )

    # -----------------------------
    # Storage API (text bytes)
    # -----------------------------
    def store_many(self, items: Dict[str, bytes]) -> StoreStats:
        if self._mem is None:
            raise RuntimeError("HolographicMemory fallback unavailable")
        t0 = time.perf_counter()
        for _id, content in items.items():
            self._mem.store(content, filename=f"{_id}.txt")
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1000.0
        count = len(items)
        th = (count / (t1 - t0)) if (t1 > t0) else 0.0
        return StoreStats(count=count, duration_ms=dur, throughput_items_per_s=th)

    def retrieve(self, doc_id: str) -> bytes:
        if self._mem is None:
            raise RuntimeError("HolographicMemory fallback unavailable")
        return self._mem.retrieve(doc_id)

    # -----------------------------
    # Resonance vector index
    # -----------------------------
    def index_vectors(self, id_to_vec: Dict[str, np.ndarray]) -> None:
        if self._res_index is None:
            raise RuntimeError("HolographicResonanceIndex fallback unavailable")
        for _id, vec in id_to_vec.items():
            self._res_index.store(text=_id, vector=np.asarray(vec, dtype=complex))

    def query_vectors(self, q: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self._res_index is None:
            raise RuntimeError("HolographicResonanceIndex fallback unavailable")
        return self._res_index.retrieve(np.asarray(q, dtype=complex), k=k)

    # -----------------------------
    # FFT ops
    # -----------------------------
    @staticmethod
    def fft1d(x: np.ndarray) -> np.ndarray:
        return np.fft.fft(np.asarray(x, dtype=complex), norm="ortho")

    @staticmethod
    def ifft1d(X: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.asarray(X, dtype=complex), norm="ortho")

    @staticmethod
    def fft3d(field: np.ndarray) -> np.ndarray:
        return np.fft.fftn(np.asarray(field, dtype=complex), norm="ortho")

    @staticmethod
    def ifft3d(spec: np.ndarray) -> np.ndarray:
        return np.fft.ifftn(np.asarray(spec, dtype=complex), norm="ortho")

    @staticmethod
    def unitary_error(U: np.ndarray) -> float:
        I = np.eye(U.shape[0], dtype=complex)
        e = np.linalg.norm(U.conj().T @ U - I, ord="fro")
        return float(e)

