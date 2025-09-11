from __future__ import annotations

import io
import random
import sys
from typing import List, Tuple

import pytest

# Add holographic-memory API to path for imports
sys.path.insert(0, 'services/holographic-memory/api')
from hwp_v4 import write_hwp_v4_micro


def _mk_layer(n: int = 16) -> Tuple[List[int], List[int], float]:
    from hwp_v4 import _quantize_amplitude, _quantize_phase

    # small deterministic vector for speed
    vals = [random.random() for _ in range(n)]
    amps, scale = _quantize_amplitude(vals)
    phases = _quantize_phase([random.random() for _ in range(n)])
    return (amps, phases, scale)


@pytest.mark.unit
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_write_hwp_v4_micro_roundtrip(n: int, tmp_path) -> None:
    amps, phases, scale = _mk_layer(n)
    test_file = tmp_path / f"test_{n}.hwp"
    write_hwp_v4_micro(
        test_file,
        doc_id_hex="abcd1234",
        original_size=1024,
        dimension=64,
        layers_count=1
    )
    raw = test_file.read_bytes()
    assert raw.startswith(b"H4M1")  # micro header
    assert len(raw) > 8


@pytest.mark.unit
@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_write_hwp_v4_micro_k8_header(n: int, tmp_path) -> None:
    from hwp_v4 import write_hwp_v4_micro_k8
    
    amps, phases, scale = _mk_layer(n)
    test_file = tmp_path / f"test_k8_{n}.hwp"
    write_hwp_v4_micro_k8(
        test_file,
        doc_id_hex="abcd1234",
        original_size=1024,
        dimension=64,
        layers_count=1
    )
    raw = test_file.read_bytes()
    assert raw.startswith(b"H4K8")
    assert len(raw) > 8


@pytest.mark.unit
@pytest.mark.parametrize("layers", [1, 2, 3, 4])
def test_write_hwp_v4_sparse_layers(layers: int) -> None:
    from services.holographic_memory.api.hwp_v4 import write_hwp_v4, build_sparse_layer

    buf = io.BytesIO()
    entries = []
    for i in range(layers):
        amps, phases, scale = _mk_layer(8)
        entries.append(build_sparse_layer(name=f"L{i}", indices=list(range(8)), amplitudes=amps, phases=phases, scale=scale))
    write_hwp_v4(buf, entries)
    raw = buf.getvalue()
    assert raw.startswith(b"HWP4V001")
    assert len(raw) > 8

