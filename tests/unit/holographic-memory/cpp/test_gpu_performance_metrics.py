from __future__ import annotations

import numpy as np
import pytest


def _gpu():
    import sys
    from pathlib import Path
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
def test_metrics_and_last_metrics_expose_core_fields():
    gpu = _gpu()
    x = np.random.rand(8, 16).astype(np.float32)
    _ = gpu.batch_encode_numpy(x, 16)
    pm = gpu.metrics()
    assert pm is not None
    # Introspect core fields expected from native struct
    for fld in ("operations_per_second", "memory_bandwidth_gb_s", "device_ms", "host_ms"):
        assert hasattr(pm, fld)
    lm = gpu.get_last_metrics()
    assert lm is not None
    for fld in ("operations_per_second", "memory_bandwidth_gb_s"):
        assert hasattr(lm, fld)
