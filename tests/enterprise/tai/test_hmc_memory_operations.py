from __future__ import annotations

import time
from typing import List, Tuple

from services.aiucp.holographic_memory.pure_python_memory import HolographicMemory


def test_store_retrieve_search_small_texts(tmp_path):
    mem = HolographicMemory()
    contents = [
        b"quantum entanglement and holography",
        b"phase retrieval using wirtinger flow",
        b"lyapunov stability and contraction fields",
        b"privacy calculus and schnorr proofs",
    ]

    ids: List[str] = []
    t0 = time.perf_counter()
    for i, c in enumerate(contents):
        ids.append(mem.store(c, filename=f"doc{i}.txt", description="test"))
    dt_store = time.perf_counter() - t0

    # Retrieve
    t1 = time.perf_counter()
    payload = mem.retrieve(ids[0])
    dt_retrieve = time.perf_counter() - t1
    assert payload == contents[0]

    # Semantic search
    q = "holography phase"
    t2 = time.perf_counter()
    results: List[Tuple[str, float]] = mem.search_semantic(q, top_k=3)
    dt_search = time.perf_counter() - t2
    assert results and results[0][0] in set(ids)

    # Performance sanity
    assert dt_store < 0.1 and dt_retrieve < 0.05 and dt_search < 0.05

