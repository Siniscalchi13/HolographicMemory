from __future__ import annotations

import numpy as np

from services.aiucp.verbum_field_engine.selection import selection_scores
from services.aiucp.holographic_memory.pure_python_memory import HolographicMemory
from services.aiucp.quantum_core.advanced_lyapunov import quadratic_stability_region
from services.aiucp.quantum_core.enhanced_privacy import gaussian_pld, pld_to_eps_delta


def test_vfe_hmc_aioc_ppc_integration():
    # VFE: select a model
    models = [
        {"name": "apple-small", "backend": "apple", "ctx": 4096, "A": [[0.2, 0.3, 0.1, 0.0, 0.5]]},
        {"name": "llama-fast", "backend": "llama.cpp", "ctx": 8192, "A": [[0.1, 0.6, 0.2, 0.0, 0.3]]},
    ]
    messages = [{"role": "user", "content": "Explain holographic memory and phase retrieval."}]
    sel, scores = selection_scores(models, messages, max_tokens=128)
    assert sel in scores

    # HMC: store/retrieve a small document
    mem = HolographicMemory()
    doc_id = mem.store(b"holographic memory content", filename="note.txt")
    payload = mem.retrieve(doc_id)
    assert payload.startswith(b"holographic")

    # AIOC: stability region for a simple stable system
    A = np.array([[-1.0, 0.2], [-0.1, -0.5]])
    reg = quadratic_stability_region(A, samples=50)
    assert reg.ok_mask.any()

    # PPC: privacy accounting for a Gaussian mechanism
    p = gaussian_pld(sigma=2.0, n=4097)
    eps = pld_to_eps_delta(p, delta=1e-6)
    assert eps >= 0.0

