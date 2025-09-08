from services.telemetry import PerformanceTelemetry


def test_track_and_suggest():
    t = PerformanceTelemetry()
    t.track_compression(1000, 100, "knowledge")
    t.track_compression(500, 50, "experience")
    orig, stored, ratio = t.current_ratios()
    assert orig == 1500
    assert stored == 150
    assert ratio and ratio > 0
    # Suggest with simple importance
    imp = {"knowledge": 1.0, "experience": 1.0}
    D = t.suggest_rebalancing(imp, total_budget=100, floors={"knowledge": 10, "experience": 10})
    assert sum(D.values()) == 100
    assert D["experience"] >= 10 and D["knowledge"] >= 10

