from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_report(summary: Dict[str, Any]) -> Dict[str, str]:
    """Write JSON and Markdown enterprise reports.

    Returns paths of written artifacts.
    """
    out_dir = os.path.join("reports", "enterprise")
    _ensure_dir(out_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(out_dir, f"test_report_{ts}.json")
    md_path = os.path.join(out_dir, f"test_report_{ts}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Render a minimal Markdown summary
    lines = []
    lines.append(f"# Enterprise Test Report ({ts})")
    lines.append("")
    if "phases" in summary:
        lines.append("## Phases")
        for phase, status in summary["phases"].items():
            lines.append(f"- {phase}: {status}")
        lines.append("")
    if "metrics" in summary:
        lines.append("## Metrics")
        for k, v in summary["metrics"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if "checks" in summary:
        lines.append("## Checks")
        for k, v in summary["checks"].items():
            lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
        lines.append("")
    if "notes" in summary and summary["notes"]:
        lines.append("## Notes")
        lines.append(str(summary["notes"]))
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json": json_path, "markdown": md_path}

