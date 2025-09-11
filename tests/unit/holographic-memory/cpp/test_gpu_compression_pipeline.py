from __future__ import annotations

import numpy as np
import pytest


def _gpu_with_params():
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
    params = hg.QuantizationParams()
    return g, params


@pytest.mark.unit
@pytest.mark.gpu
def test_quantize_and_reconstruct():
    gpu, params = _gpu_with_params()
    n, d = 32, 32
    real = np.random.rand(n, d).astype(np.float32)
    imag = np.random.rand(n, d).astype(np.float32)
    # Quantize (with validation path)
    q = gpu.gpu_holographic_quantize_with_validation(real, imag, 0, params)
    assert q is not None
    # Basic reconstruction call (phase is arbitrary random for test)
    phase = np.random.rand(n, d).astype(np.float32)
    rec = gpu.gpu_holographic_wave_reconstruction(real.tolist(), imag.tolist(), phase.tolist(), 0)
    assert rec is not None


@pytest.mark.unit
@pytest.mark.gpu
def test_quantization_statistics():
    gpu, _ = _gpu_with_params()
    # Create 2D arrays as expected by the C++ function
    errs1 = np.random.rand(8, 8).astype(np.float32)
    errs2 = np.random.rand(8, 8).astype(np.float32)
    stats = gpu.gpu_quantization_statistics(errs1.tolist(), errs2.tolist())
    assert stats is not None
