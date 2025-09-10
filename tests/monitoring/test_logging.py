from __future__ import annotations

import logging
import pytest


@pytest.mark.monitoring
def test_logging_capture(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger(__name__).info("test message")
    assert any("test message" in rec.message for rec in caplog.records)

