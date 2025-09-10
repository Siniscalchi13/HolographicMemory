from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

from .test_logger import EnterpriseTestLogger


@dataclass
class _SessionState:
    logger: EnterpriseTestLogger
    results: List[Dict[str, Any]]
    html_output_dir: Path
    logs_dir: Path


def _load_test_config() -> Dict[str, Any]:
    cfg = Path("tests/config/test_config.yaml")
    if not cfg.exists():
        return {}
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(cfg.read_text(encoding="utf-8")) or {})
    except Exception:
        return {}


def _ensure_report_dirs() -> tuple[Path, Path, Path]:
    base = Path("tests/reports")
    logs = base / "logs"
    html = base / "html"
    json_dir = base / "json"
    for d in (logs, html, json_dir):
        d.mkdir(parents=True, exist_ok=True)
    return logs, html, json_dir


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: D401
    cfg = _load_test_config()
    logs_dir, html_dir, _ = _ensure_report_dirs()
    suite_name = cfg.get("test_suite", {}).get("name", "HolographicMemory Enterprise Test Suite")
    session._enterprise_state = _SessionState(  # type: ignore[attr-defined]
        logger=EnterpriseTestLogger(test_name=suite_name, log_dir=logs_dir),
        results=[],
        html_output_dir=html_dir,
        logs_dir=logs_dir,
    )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:  # type: ignore[override]
    # Only act on call phase for outcome
    if report.when != "call":
        return
    # Fetch state from session
    session = report._session
    state: _SessionState = getattr(session, "_enterprise_state", None)  # type: ignore[attr-defined]
    if not state:
        return
    entry = {
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "duration": getattr(report, "duration", 0.0),
        "longrepr": str(report.longrepr) if report.failed else "",
    }
    state.results.append(entry)
    # Stream to execution log
    state.logger.test_logger.info(json.dumps({
        "event": "test_result",
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "duration": entry["duration"],
    }))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: D401
    state: _SessionState = getattr(session, "_enterprise_state", None)  # type: ignore[attr-defined]
    if not state:
        return
    # Write JSON results
    json_dir = Path("tests/reports/json")
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / "test_results.json").write_text(json.dumps(state.results, indent=2), encoding="utf-8")
    # Optionally generate HTML dashboards if Jinja2 available
    try:
        from tests.reports.html_generator import HTMLReportGenerator

        gen = HTMLReportGenerator(log_dir=state.logs_dir, output_dir=state.html_output_dir)
        gen.generate_all_reports()
    except Exception:
        # Non-fatal; pytest-html still produces a basic report
        pass

