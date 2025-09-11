from __future__ import annotations

import logging

import pytest


@pytest.mark.monitoring
@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
def test_log_levels_capture(caplog, level):  # type: ignore
    logger = logging.getLogger("holo.test")
    with caplog.at_level(level):
        logger.debug("dbg")
        logger.info("inf")
        logger.warning("warn")
        logger.error("err")
    # all emitted up to level should be present at least once
    assert any(r.levelno >= level for r in caplog.records)

