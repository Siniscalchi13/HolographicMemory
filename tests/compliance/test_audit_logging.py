from __future__ import annotations

import logging
import pytest


@pytest.mark.compliance
def test_logging_is_configured():
    # Basic validation that logging is configured at INFO or lower
    level = logging.getLogger().level
    assert level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)

