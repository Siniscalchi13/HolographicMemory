from __future__ import annotations

from pathlib import Path
import json


def load_coverage_json(path: str = "tests/reports/coverage/coverage.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8") or "{}")


def summary(cov: dict) -> dict:
    files = cov.get("files", {})
    totals = cov.get("totals", {})
    return {
        "file_count": len(files),
        "percent_covered": totals.get("percent_covered", 0.0),
        "num_statements": totals.get("num_statements", 0),
        "num_missing": totals.get("num_missing", 0),
    }


if __name__ == "__main__":
    cov = load_coverage_json()
    print(summary(cov))

