from __future__ import annotations

from pathlib import Path

from .test_logger import EnterpriseTestLogger


def get_security_logger(test_name: str, base_dir: Path) -> EnterpriseTestLogger:
    return EnterpriseTestLogger(test_name=test_name, log_dir=base_dir)

