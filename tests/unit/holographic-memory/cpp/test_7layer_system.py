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
def test_initialize_and_layer_stats():
    gpu = _gpu()
    # Initialize with small budget (returns None but sets internal state)
    ok = gpu.initialize_7layer_decomposition(64)
    # Function returns None but initializes internal state
    # Flag should be true after init
    assert bool(getattr(gpu, 'layers_initialized', False)) is True
    stats = gpu.get_layer_stats()
    assert isinstance(stats, dict)


@pytest.mark.unit
@pytest.mark.gpu
def test_optimize_and_snr_paths():
    gpu = _gpu()
    gpu.initialize_7layer_decomposition(64)
    # Optimize dimensions and update SNRs
    _ = gpu.optimize_layer_dimensions()
    _ = gpu.update_layer_snrs()
    # Calculate for a few layers
    snr0 = gpu.calculate_layer_snr(0)
    snr1 = gpu.calculate_layer_snr(1)
    assert snr0 is not None and snr1 is not None
