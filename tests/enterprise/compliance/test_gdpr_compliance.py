from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.compliance
def test_audit_logging_event_recorded(tmp_path: Path):
    # Smoke: ensure we can write an audit event via logging utility
    from tests.logging.test_logger import EnterpriseTestLogger

    logger = EnterpriseTestLogger(test_name="gdpr", log_dir=Path("tests/reports/logs"))
    logger.log_compliance_event("GDPR", "Data Minimization", "pass", {"note": "no pii"})
    assert (Path("tests/reports/logs") / "compliance.log").exists()

