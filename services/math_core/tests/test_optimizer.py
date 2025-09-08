from services.math_core import DimensionOptimizer


def test_optimize_dimensions_sums_to_budget():
    opt = DimensionOptimizer()
    loads = {"a": 100, "b": 50, "c": 25}
    importance = {"a": 1.0, "b": 1.0, "c": 1.0}
    D = opt.optimize_dimensions(loads, importance, total_budget=407)
    assert sum(D.values()) == 407
    # sanity: more load should yield fewer dims ceteris paribus
    assert D["a"] < D["b"] < D["c"]

