from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader


class HTMLReportGenerator:
    """Enterprise-grade HTML report generator using Jinja2 templates."""

    def __init__(self, log_dir: Path, output_dir: Path):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        (self.output_dir / "assets").mkdir(parents=True, exist_ok=True)

    def generate_all_reports(self) -> None:
        self.generate_main_dashboard()
        self.generate_test_execution_report()
        self.generate_performance_report()
        self.generate_security_report()
        self.generate_compliance_report()
        self.generate_coverage_report()
        self.copy_assets()

    # --------- Main dashboard ---------
    def generate_main_dashboard(self) -> None:
        t = self.env.get_template("dashboard.html")
        data = {
            "test": self._collect_test_data(),
            "performance": self._collect_performance_data(),
        }
        security = self._collect_security_data()
        compliance = self._collect_compliance_data()
        coverage = self._collect_coverage_data()
        summary = self._summary(data["test"], data["performance"], security, compliance, coverage)
        html = t.render(
            summary=summary,
            test_data=data["test"],
            performance_data=data["performance"],
            security_data=security,
            compliance_data=compliance,
            coverage_data=coverage,
            timestamp=datetime.utcnow().isoformat(),
            title="HolographicMemory Enterprise Test Suite Report",
        )
        (self.output_dir / "index.html").write_text(html, encoding="utf-8")

    # --------- Section reports ---------
    def generate_test_execution_report(self) -> None:
        t = self.env.get_template("test_execution.html")
        logs = self._parse_json_lines(self.log_dir / "test_execution.log")
        html = t.render(logs=logs, timestamp=datetime.utcnow().isoformat())
        (self.output_dir / "test_execution.html").write_text(html, encoding="utf-8")

    def generate_performance_report(self) -> None:
        t = self.env.get_template("performance_analysis.html")
        logs = self._parse_json_lines(self.log_dir / "performance.log")
        html = t.render(logs=logs, timestamp=datetime.utcnow().isoformat())
        (self.output_dir / "performance_analysis.html").write_text(html, encoding="utf-8")

    def generate_security_report(self) -> None:
        t = self.env.get_template("security_assessment.html")
        logs = self._parse_json_lines(self.log_dir / "security.log")
        html = t.render(logs=logs, timestamp=datetime.utcnow().isoformat())
        (self.output_dir / "security_assessment.html").write_text(html, encoding="utf-8")

    def generate_compliance_report(self) -> None:
        t = self.env.get_template("compliance_report.html")
        logs = self._parse_json_lines(self.log_dir / "compliance.log")
        html = t.render(logs=logs, timestamp=datetime.utcnow().isoformat())
        (self.output_dir / "compliance_report.html").write_text(html, encoding="utf-8")

    def generate_coverage_report(self) -> None:
        t = self.env.get_template("coverage_report.html")
        cov = self._collect_coverage_data()
        html = t.render(coverage=cov, timestamp=datetime.utcnow().isoformat())
        (self.output_dir / "coverage_report.html").write_text(html, encoding="utf-8")

    # --------- Helpers ---------
    def _collect_test_data(self) -> Dict[str, Any]:
        results_path = Path("tests/reports/json/test_results.json")
        if results_path.exists():
            try:
                return {"tests": json.loads(results_path.read_text(encoding="utf-8"))}
            except Exception:
                return {"tests": []}
        return {"tests": []}

    def _collect_performance_data(self) -> Dict[str, Any]:
        return {"metrics": self._parse_json_lines(self.log_dir / "performance.log")}

    def _collect_security_data(self) -> Dict[str, Any]:
        return {"events": self._parse_json_lines(self.log_dir / "security.log")}

    def _collect_compliance_data(self) -> Dict[str, Any]:
        return {"validations": self._parse_json_lines(self.log_dir / "compliance.log")}

    def _collect_coverage_data(self) -> Dict[str, Any]:
        cov_json = Path("tests/reports/coverage/coverage.json")
        if cov_json.exists():
            try:
                return json.loads(cov_json.read_text())
            except Exception:
                return {"coverage": 0}
        return {"coverage": 0}

    @staticmethod
    def _parse_json_lines(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        items: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        return items

    @staticmethod
    def _summary(test: Dict[str, Any], perf: Dict[str, Any], sec: Dict[str, Any], comp: Dict[str, Any], cov: Dict[str, Any]) -> Dict[str, Any]:
        tests = test.get("tests", [])
        return {
            "total_tests": len(tests),
            "passed": sum(1 for t in tests if t.get("outcome") == "passed"),
            "failed": sum(1 for t in tests if t.get("outcome") == "failed"),
            "perf_events": len(perf.get("metrics", [])),
            "security_events": len(sec.get("events", [])),
            "compliance_events": len(comp.get("validations", [])),
            "coverage": cov.get("totals", {}).get("percent_covered", cov.get("coverage", 0)),
        }

    def copy_assets(self) -> None:
        css = (
            "body{font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:20px;}"
            ".container{max-width:1200px;margin:0 auto;background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1);}"
            ".header{border-bottom:2px solid #007acc;padding-bottom:10px;margin-bottom:20px;}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;}"
            ".card{background:#f8f9fa;padding:12px;border-radius:6px;border-left:4px solid #007acc;}"
        )
        (self.output_dir / "assets/style.css").write_text(css, encoding="utf-8")
