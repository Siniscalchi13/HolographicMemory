from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class TheoryPredictions:
    fft_unitarity_err_tol: float
    chsh_tsirelson: float
    retrieval_expected_accuracy: float  # heuristic bound for synthetic data


class TheoryService:
    """
    Provides mathematical expectations to compare against actual outcomes.
    - FFT unitarity tolerance (numerical)
    - CHSH Tsirelson bound
    - Retrieval accuracy estimate given synthetic generation scheme
    """

    def __init__(self, n: int) -> None:
        self.n = int(n)

    def predictions(self) -> TheoryPredictions:
        # Empirical tolerance for unitary DFT with ortho-norm (double precision)
        tol = 1e-10 if self.n <= 4096 else 1e-8
        # Tsirelson bound for CHSH
        ts = 2.0 * math.sqrt(2.0)
        # For our synthetic data, text→state is separable and distinct; with
        # resonance indexing we expect ≥0.9 top-1 for exact texts
        acc = 0.9
        return TheoryPredictions(fft_unitarity_err_tol=tol, chsh_tsirelson=ts, retrieval_expected_accuracy=acc)

    @staticmethod
    def fft_unitarity_error(n: int) -> float:
        # Construct DFT matrix and check U*U≈I; only for small n
        U = np.fft.fft(np.eye(n), norm="ortho")
        I = np.eye(n, dtype=complex)
        e = np.linalg.norm(U.conj().T @ U - I, ord="fro")
        return float(e)

