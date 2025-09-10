from __future__ import annotations

import pytest


@pytest.mark.unit
def test_telemetry_ratios_and_snapshot():
    from services.telemetry.performance_tracker import PerformanceTelemetry

    t = PerformanceTelemetry()
    t.track_compression(original=1000, stored=500, layer="knowledge")
    t.track_compression(original=2000, stored=1000, layer="context")
    orig, stored, ratio = t.current_ratios()
    assert orig == 3000 and stored == 1500
    assert ratio == pytest.approx(2.0, abs=1e-6)
    snap = t.snapshot()
    assert snap["overall"]["compression_x"] == pytest.approx(2.0, abs=1e-6)
    assert set(snap["per_layer"]) >= {"knowledge", "context"}

