from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "size,expected",
    [
        (0, 8),
        (1, 8),
        (511, 8),
        (512, 16),
        (4095, 16),
        (4096, 32),
        (32767, 32),
        (32768, 64),
        (262143, 64),
        (262144, 128),
        (1048575, 128),
        (1048576, 256),
        (10_000_000, 256),
    ],
)
def test_calculate_optimal_dimension(size: int, expected: int) -> None:
    from services.holographic_memory.core.holographicfs.memory import calculate_optimal_dimension

    assert calculate_optimal_dimension(size) == expected

