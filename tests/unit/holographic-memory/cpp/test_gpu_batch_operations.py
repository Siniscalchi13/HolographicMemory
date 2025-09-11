from __future__ import annotations

import numpy as np
import pytest


def _gpu():
    import sys
    from pathlib import Path

    # Ensure native libs are on path (in case tests run standalone)
    root = Path('services/holographic-memory/core/native/holographic')
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and (d.name == 'build' or d.name.startswith('lib.')):
                p = str(d.resolve())
                if p not in sys.path:
                    sys.path.insert(0, p)
    import holographic_gpu as hg

    g = hg.HolographicGPU()
    try:
        g.initialize('metal')
    except Exception:
        try:
            g.initialize()
        except Exception:
            pass
    return g


@pytest.mark.unit
@pytest.mark.gpu
def test_batch_encode_numpy_returns_per_item_vectors():
    gpu = _gpu()
    x = np.random.rand(16, 32).astype(np.float32)
    out = gpu.batch_encode_numpy(x, 32)
    assert out is not None
    assert isinstance(out, list)
    assert len(out) == x.shape[0]


@pytest.mark.unit
@pytest.mark.gpu
def test_batch_encode_python_list_roundtrip_shape():
    gpu = _gpu()
    # simple list-of-lists input
    batch = [[float(i + j) for j in range(16)] for i in range(8)]
    out = gpu.batch_encode(batch, 16)
    assert out is not None
    assert isinstance(out, list)
    assert len(out) == len(batch)
