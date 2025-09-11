from __future__ import annotations

import io
from pathlib import Path

import pytest


def _mk_layer(n: int = 16):
    import sys
    sys.path.insert(0, 'services/holographic-memory/api')
    from hwp_v4 import _quantize_amplitude, _quantize_phase
    import random

    vals = [random.random() for _ in range(n)]
    amps, scale = _quantize_amplitude(vals)
    phases = _quantize_phase([random.random() for _ in range(n)])
    return list(range(n)), amps, phases, scale


@pytest.mark.integration
def test_write_micro_hwp_to_disk(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, 'services/holographic-memory/api')
    from hwp_v4 import write_hwp_v4_micro

    p = tmp_path / "micro.hwp"
    write_hwp_v4_micro(
        p,
        doc_id_hex="abcd1234",
        original_size=1024,
        dimension=64,
        layers_count=1
    )
    raw = p.read_bytes()
    assert raw.startswith(b"H4M1")
    assert p.stat().st_size > 8


@pytest.mark.integration
def test_write_sparse_hwp_v4_to_disk(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, 'services/holographic-memory/api')
    from hwp_v4 import write_hwp_v4, build_sparse_layer

    # Create test layers
    layers = []
    for i in range(2):
        idx, amps, phases, scale = _mk_layer(16)
        layers.append(build_sparse_layer(name=f"L{i}", amplitudes=amps, phases=phases, top_k=16))
    
    p = tmp_path / "sparse.hwp"
    write_hwp_v4(
        p,
        doc_id="test123",
        filename="test.txt",
        original_size=1024,
        content_type="text/plain",
        dimension=64,
        layers=layers
    )
    raw = p.read_bytes()
    assert raw.startswith(b"HWP4V001")
    assert p.stat().st_size > 8

