from __future__ import annotations

import json
from pathlib import Path


def test_alerting_rules_yaml_present():
    p = Path("docs/monitoring/alerting-rules.yml")
    assert p.is_file()


def test_grafana_dashboards_have_core_panels():
    # Spot-check presence of key panel titles/queries
    qec = json.loads(Path("docs/monitoring/grafana-qec-dashboard.json").read_text(encoding="utf-8"))
    titles = {panel.get("title") for panel in qec.get("panels", [])}
    assert {"QEC Requests (Success)", "QEC Requests (Errors)"}.issubset(titles)

