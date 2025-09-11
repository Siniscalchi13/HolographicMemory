from __future__ import annotations

from typing import Dict

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "layer,orig,stored",
    [
        ("Identity", 100, 50),
        ("Knowledge", 200, 100),
        ("Experience", 300, 150),
        ("Preference", 400, 200),
        ("Context", 500, 250),
        ("Wisdom", 128, 64),
        ("Vault", 64, 32),
        ("Identity", 0, 0),
        ("Context", 123, 0),
        ("Preference", 0, 123),
    ],
)
def test_track_and_snapshot(layer: str, orig: int, stored: int) -> None:
    from services.telemetry.performance_tracker import PerformanceTelemetry

    t = PerformanceTelemetry()
    t.track_compression(orig, stored, layer)
    t.track_retrieval()
    snap = t.snapshot()
    assert "overall" in snap and "per_layer" in snap
    assert isinstance(snap["overall"].get("retrievals", 0), int)
    # per-layer stats include the layer we tracked
    assert layer in snap["per_layer"] or stored == 0 == orig


@pytest.mark.unit
@pytest.mark.parametrize(
    "loads,importance,budget",
    [
        ({"Identity": 1, "Knowledge": 2, "Vault": 1}, {"Identity": 1.0, "Knowledge": 0.8, "Vault": 0.2}, 64),
        ({"Identity": 10, "Knowledge": 10, "Vault": 10}, {"Identity": 0.5, "Knowledge": 0.4, "Vault": 0.1}, 1024),
        ({"A": 3, "B": 5, "C": 7}, {"A": 1.0, "B": 2.0, "C": 3.0}, 333),
        ({"Only": 1}, {"Only": 1.0}, 11),
    ],
)
def test_suggest_rebalancing(loads: Dict[str, int], importance: Dict[str, float], budget: int) -> None:
    from services.telemetry.performance_tracker import PerformanceTelemetry

    t = PerformanceTelemetry()
    # seed loads to telemetry to exercise importance weighting effect
    for k, n in loads.items():
        for _ in range(max(1, n)):
            t.track_compression(10, 5, k)
    D = t.suggest_rebalancing(importance=importance, total_budget=budget, floors=None)
    assert sum(D.values()) == int(budget)
    assert set(D.keys()) == set(loads.keys())


@pytest.mark.unit
@pytest.mark.parametrize(
    "entries",
    [
        [("Identity", 10, 5), ("Knowledge", 20, 10), ("Wisdom", 5, 2)],
        [("Context", 100, 50), ("Preference", 5, 1)],
        [("Experience", 7, 7), ("Vault", 32, 8), ("Knowledge", 64, 16)],
        [("Identity", 0, 0), ("Wisdom", 1, 0)],
        [("Identity", 123, 61), ("Context", 321, 80), ("Preference", 222, 111), ("Experience", 3, 1)],
        [("A", 1, 1), ("B", 2, 1), ("C", 3, 1)],
        [("A", 1000, 100), ("B", 1, 1), ("C", 10, 1)],
    ],
)
def test_multiple_track_entries(entries):
    from services.telemetry.performance_tracker import PerformanceTelemetry

    t = PerformanceTelemetry()
    for layer, o, s in entries:
        t.track_compression(o, s, layer)
    snap = t.snapshot()
    assert isinstance(snap, dict)
    assert "per_layer" in snap
