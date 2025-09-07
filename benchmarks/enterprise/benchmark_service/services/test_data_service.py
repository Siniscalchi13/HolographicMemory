from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ..contracts import DataSpec, Dataset, TextItem

try:
    from services.aiucp.quantum_core.mathematical_implementation import QuantumMathCore
except Exception:
    QuantumMathCore = None  # type: ignore


class TestDataService:
    """
    Synthetic dataset generation for benchmarking.

    - Generates text corpus with IDs
    - Encodes deterministic complex state vectors (QuantumMathCore)
    - Generates 3D complex fields for FFT/storage tests
    """

    def __init__(self, spec: DataSpec) -> None:
        self.spec = spec
        self._rng = np.random.default_rng(spec.seed)
        n = spec.vector_dim
        self._qm = QuantumMathCore(dimension=n) if QuantumMathCore else None

    def _texts(self) -> List[TextItem]:
        base = [
            "a hologram of memory",
            "quantum phase retrieval",
            "fftw and apple accelerate",
            "retrieval with exact recall",
            "chsh and tsirelson bound",
            "faiss vector index",
            "spatial allocation in 3d",
            "complex field simulation",
            "embeddings to storage pipeline",
            "semantic search comparison",
        ]
        out: List[TextItem] = []
        m = max(1, (self.spec.num_items // len(base)) + 1)
        i = 0
        for _ in range(m):
            for t in base:
                if i >= self.spec.num_items:
                    break
                out.append(TextItem(id=f"doc-{i:04d}", text=f"{t} #{i}"))
                i += 1
        return out[: self.spec.num_items]

    def _encode(self, text: str) -> np.ndarray:
        if self._qm is not None:
            return self._qm.encode_text_state(text, self.spec.vector_dim)
        # Fallback: simple hash embedding into complex unit vector
        n = self.spec.vector_dim
        real = np.zeros(n, dtype=float)
        imag = np.zeros(n, dtype=float)
        for i in range(0, len(text)):
            h = hash(text[i : i + 3])
            idx = (h % n + n) % n
            phase = (h & 0xFFFF) / 0xFFFF * 2 * np.pi
            real[idx] += np.cos(phase)
            imag[idx] += np.sin(phase)
        v = real + 1j * imag
        nrm = np.linalg.norm(v)
        return (v / nrm) if nrm > 0 else np.ones(n, dtype=complex) / np.sqrt(n)

    def _field3d(self) -> np.ndarray:
        z, y, x = self.spec.field_shape
        # random complex field with smoothness along axes
        a = self._rng.normal(0, 1, size=(z, y, x))
        b = self._rng.normal(0, 1, size=(z, y, x))
        field = a + 1j * b
        # low-pass filter in Fourier to create structured fields
        F = np.fft.fftn(field)
        kz = np.fft.fftfreq(z)[:, None, None]
        ky = np.fft.fftfreq(y)[None, :, None]
        kx = np.fft.fftfreq(x)[None, None, :]
        mask = (kz**2 + ky**2 + kx**2) < 0.15
        F = F * mask
        smooth = np.fft.ifftn(F)
        return smooth.astype(complex)

    def generate(self) -> Dataset:
        texts = self._texts()
        states: Dict[str, np.ndarray] = {t.id: self._encode(t.text) for t in texts}
        fields3d: Dict[str, np.ndarray] = {t.id: self._field3d() for t in texts[: min(8, len(texts))]}
        return Dataset(spec=self.spec, texts=texts, states=states, fields3d=fields3d)

