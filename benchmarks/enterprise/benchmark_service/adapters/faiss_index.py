from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FaissInfo:
    available: bool
    dim: int
    index_type: str


class FaissIndex:
    """
    Thin wrapper around FAISS (if available). Converts complex vectors to real
    by concatenating real and imaginary parts.
    """

    def __init__(self, dim_complex: int) -> None:
        self.dim_complex = int(dim_complex)
        self._faiss = None
        self._index = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dim_complex * 2)
        except Exception:
            self._faiss = None
            self._index = None

    @staticmethod
    def _c2r(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=complex).reshape(-1)
        return np.concatenate([v.real, v.imag]).astype("float32")

    def info(self) -> FaissInfo:
        return FaissInfo(available=self._index is not None, dim=self.dim_complex, index_type="IndexFlatIP")

    def add(self, vecs: List[np.ndarray]) -> None:
        if self._index is None:
            raise RuntimeError("FAISS not available")
        mat = np.stack([self._c2r(v) for v in vecs], axis=0)
        self._index.add(mat)

    def search(self, q: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if self._index is None:
            raise RuntimeError("FAISS not available")
        qv = self._c2r(q)[None, :]
        D, I = self._index.search(qv, k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        return list(zip(idxs, scores))

