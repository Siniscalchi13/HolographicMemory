from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_calculate_optimal_dimension_thresholds():
    import sys
    sys.path.insert(0, 'services/holographic-memory/core')
    from holographicfs.memory import calculate_optimal_dimension

    assert calculate_optimal_dimension(1) == 8
    assert calculate_optimal_dimension(1024) == 16
    assert calculate_optimal_dimension(10_000) == 32
    assert calculate_optimal_dimension(200_000) == 64
    assert calculate_optimal_dimension(900_000) == 128
    assert calculate_optimal_dimension(10_000_000) == 256


@pytest.mark.unit
def test_get_wave_data_from_bytes_shapes():
    from holographicfs.memory import Memory

    # Use Memory utility that only touches numpy path for get_wave_data_from_bytes
    m = Memory.__new__(Memory)  # bypass heavy init
    raw = bytes(range(0, 200))
    wave = m.get_wave_data_from_bytes(raw, doc_id="d1")
    assert wave["dimension"] > 0
    assert len(wave["amplitudes"]) == wave["dimension"]
    assert len(wave["phases"]) == wave["dimension"]

