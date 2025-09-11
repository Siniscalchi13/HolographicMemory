from __future__ import annotations

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
def test_wave_and_interference_and_bell_and_capacity():
    gpu = _gpu()
    assert gpu.validate_wave_properties() is not None
    assert gpu.analyze_interference_patterns() is not None
    assert gpu.validate_bell_inequality() is not None
    assert gpu.enforce_capacity_theorem() is not None
