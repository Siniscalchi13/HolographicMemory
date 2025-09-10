from __future__ import annotations

import json
from pathlib import Path


def test_helm_charts_presence_and_yaml():
    # Basic presence checks
    charts = [
        Path("charts/vfe/Chart.yaml"),
        Path("charts/hmc/Chart.yaml"),
        Path("charts/aiocp/Chart.yaml"),
        Path("charts/observability/Chart.yaml"),
    ]
    for p in charts:
        assert p.is_file(), f"missing chart: {p}"


def test_monitoring_dashboards_json_parses():
    paths = [
        Path("docs/monitoring/grafana-vfe-dashboard.json"),
        Path("docs/monitoring/grafana-hmc-dashboard.json"),
        Path("docs/monitoring/grafana-qec-dashboard.json"),
    ]
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "title" in data and "panels" in data

