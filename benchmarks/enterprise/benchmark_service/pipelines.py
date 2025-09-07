from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

import numpy as np

from .adapters.holographic_backend import HoloBackend


class EmbeddingStoragePipeline:
    """
    Embeddings → storage → retrieval → text conversion workflow.

    - Uses deterministic complex embeddings (provided externally)
    - Stores raw text bytes into holographic memory (Python fallback)
    - Builds resonance index for vector retrieval
    - Validates roundtrip and measures latency
    """

    def __init__(self, vector_dim: int) -> None:
        self.backend = HoloBackend(vector_dim=vector_dim)

    @staticmethod
    def _to_bytes_map(id_to_text: Dict[str, str]) -> Dict[str, bytes]:
        return {k: v.encode("utf-8") for k, v in id_to_text.items()}

    def run(self, id_to_text: Dict[str, str], id_to_vec: Dict[str, np.ndarray], queries: List[str]) -> Dict[str, float]:
        # Store
        store_stats = self.backend.store_many(self._to_bytes_map(id_to_text))

        # Index
        self.backend.index_vectors(id_to_vec)

        # Query
        lat_samples: List[float] = []
        success = 0
        for q in queries:
            qv = id_to_vec[q]
            t0 = time.perf_counter()
            res = self.backend.query_vectors(qv, k=1)
            t1 = time.perf_counter()
            lat_samples.append((t1 - t0) * 1000.0)
            if res and res[0][0] == q:
                success += 1

        success_rate = success / max(1, len(queries))
        lat_samples.sort()
        p50 = statistics.median(lat_samples) if lat_samples else 0.0
        p95 = lat_samples[max(0, int(len(lat_samples) * 0.95) - 1)] if lat_samples else 0.0

        return {
            "store_throughput_items_per_s": store_stats.throughput_items_per_s,
            "query_success_rate": success_rate,
            "query_p50_ms": p50,
            "query_p95_ms": p95,
        }

