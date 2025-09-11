from __future__ import annotations

import math
import sys
from typing import Dict

import pytest

# Add math-core to path for imports
sys.path.insert(0, 'services/math-core')
from optimizer import DimensionOptimizer


@pytest.mark.unit
@pytest.mark.parametrize(
    "loads,importance,budget",
    [
        ({"a": 1, "b": 1, "c": 1}, {"a": 1.0, "b": 1.0, "c": 1.0}, 0),
        ({"a": 0, "b": 0, "c": 0}, {"a": 1.0, "b": 2.0, "c": 3.0}, 1024),
        ({"a": 10, "b": 1, "c": 1}, {"a": 1.0, "b": 1.0, "c": 1.0}, 1024),
        ({"a": 1, "b": 10, "c": 1}, {"a": 1.0, "b": 1.0, "c": 1.0}, 1024),
        ({"a": 1, "b": 1, "c": 10}, {"a": 1.0, "b": 1.0, "c": 1.0}, 1024),
        ({"a": 5, "b": 7, "c": 11}, {"a": 0.1, "b": 1.0, "c": 10.0}, 777),
        ({"a": 100, "b": 1, "c": 1}, {"a": 10.0, "b": 10.0, "c": 10.0}, 9999),
        ({"a": 3, "b": 3, "c": 3}, {"a": 0.0, "b": 1.0, "c": 0.0}, 33),
        ({"a": 2, "b": 4, "c": 8, "d": 16}, {"a": 1, "b": 2, "c": 3, "d": 4}, 4096),
        ({"Identity": 1, "Knowledge": 2, "Experience": 3, "Preference": 4, "Context": 5, "Wisdom": 6, "Vault": 7},
         {"Identity": 1.0, "Knowledge": 0.9, "Experience": 0.8, "Preference": 0.7, "Context": 0.6, "Wisdom": 0.5, "Vault": 0.4}, 1024),
    ],
)
def test_optimize_dimensions_budget_normalization(loads: Dict[str, int], importance: Dict[str, float], budget: int) -> None:
    D = DimensionOptimizer().optimize_dimensions(loads, importance, budget, floors=None)
    assert isinstance(D, dict)
    assert sum(D.values()) == int(budget) if budget > 0 else sum(D.values()) == 0
    # Non-negative allocations
    assert all(v >= 0 for v in D.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    "floors,budget",
    [
        ({"a": 10, "b": 0, "c": 0}, 10),
        ({"a": 5, "b": 5, "c": 5}, 30),
        ({"a": 100, "b": 100, "c": 100}, 150),  # infeasible: must scale down
        ({"Identity": 64, "Knowledge": 32, "Vault": 1}, 97),
        ({"Identity": 0, "Knowledge": 0, "Vault": 0}, 0),
        ({"x": 1, "y": 1, "z": 1}, 2),  # rounding/normalization path
    ],
)
def test_optimize_dimensions_floors(loads_example: Dict[str, int], floors: Dict[str, int], budget: int) -> None:
    loads = dict(loads_example)
    # Ensure all keys present in loads
    for k in floors:
        loads.setdefault(k, 1)
    D = DimensionOptimizer().optimize_dimensions(loads, {k: 1.0 for k in loads}, budget, floors=floors)
    assert sum(D.values()) == int(budget)
    for k, f in floors.items():
        assert D.get(k, 0) >= max(0, int(f)) or budget < sum(floors.values())


@pytest.fixture(scope="module")
def loads_example() -> Dict[str, int]:
    return {"a": 3, "b": 5, "c": 7}


@pytest.mark.unit
@pytest.mark.parametrize(
    "importance",
    [
        {"a": 1.0, "b": 1.0, "c": 1.0},
        {"a": 0.0, "b": 1.0, "c": 0.0},
        {"a": 10.0, "b": 1.0, "c": 0.1},
        {"a": 0.5, "b": 0.5, "c": 0.5},
        {"a": 3.14, "b": 2.71, "c": 1.61},
        {"a": 1e-6, "b": 1e-3, "c": 1e-1},
        {"a": 100.0, "b": 0.0, "c": 0.0},
        {"a": 0.0, "b": 100.0, "c": 0.0},
        {"a": 0.0, "b": 0.0, "c": 100.0},
        {"a": 2.0, "b": 4.0, "c": 8.0},
    ],
)
def test_relative_allocation_trends(loads_example: Dict[str, int], importance: Dict[str, float]) -> None:

    D = DimensionOptimizer().optimize_dimensions(loads_example, importance, 1000)
    # Higher importance should not yield zero when budget is ample
    if importance.get("a", 0) > 0:
        assert D["a"] > 0
    if importance.get("b", 0) > 0:
        assert D["b"] > 0
    if importance.get("c", 0) > 0:
        assert D["c"] > 0


@pytest.mark.unit
@pytest.mark.parametrize("budget", [1, 2, 3, 5, 10, 33, 64, 127, 255, 1024, 2048, 4096])
def test_normalization_exact_sum(loads_example: Dict[str, int], budget: int) -> None:

    D = DimensionOptimizer().optimize_dimensions(loads_example, {k: 1.0 for k in loads_example}, budget)
    assert sum(D.values()) == int(budget)


@pytest.mark.unit
@pytest.mark.parametrize(
    "loads",
    [
        {"a": 0, "b": 0, "c": 1},
        {"a": 1, "b": 0, "c": 0},
        {"a": 0, "b": 1, "c": 0},
        {"a": 1000, "b": 1, "c": 1},
        {"a": 1, "b": 1000, "c": 1},
        {"a": 1, "b": 1, "c": 1000},
        {"a": 999999, "b": 888888, "c": 777777},
        {"a": 2**20, "b": 2**19, "c": 2**18},
        {"x": 3, "y": 5, "z": 7, "t": 11, "u": 13},
        {"Identity": 1, "Knowledge": 1, "Experience": 1, "Preference": 1, "Context": 1, "Wisdom": 1, "Vault": 1},
    ],
)
def test_extreme_loads_stable(loads: Dict[str, int]) -> None:

    D = DimensionOptimizer().optimize_dimensions(loads, {k: 1.0 for k in loads}, 1234)
    # Ensure mapping includes all keys and values are integers >= 0
    assert set(D.keys()) == set(loads.keys())
    assert all(isinstance(v, int) and v >= 0 for v in D.values())

