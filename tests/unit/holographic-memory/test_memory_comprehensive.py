from __future__ import annotations

import io
import random
from typing import List, Tuple

import pytest


def _mk_layer(n: int = 16) -> Tuple[List[int], List[int], float]:
    from services.holographic_memory.api.hwp_v4 import _quantize_amplitude, _quantize_phase

    # small deterministic vector for speed
    vals = [random.random() for _ in range(n)]
    amps, scale = _quantize_amplitude(vals)
    phases = _quantize_phase([random.random() for _ in range(n)])
    return (amps, phases, scale)


@pytest.mark.unit
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_write_hwp_v4_micro_roundtrip(n: int) -> None:
    from services.holographic_memory.api.hwp_v4 import write_hwp_v4_micro

    buf = io.BytesIO()
    amps, phases, scale = _mk_layer(n)
    write_hwp_v4_micro(buf, list(range(n)), amps, phases, scale)
    raw = buf.getvalue()
    assert raw.startswith(b"H4M1")  # micro header
    assert len(raw) > 8


@pytest.mark.unit
@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_write_hwp_v4_micro_k8_header(n: int) -> None:
    from services.holographic_memory.api.hwp_v4 import write_hwp_v4_micro_k8

    buf = io.BytesIO()
    amps, phases, scale = _mk_layer(n)
    write_hwp_v4_micro_k8(buf, list(range(n)), amps, phases, scale)
    raw = buf.getvalue()
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

