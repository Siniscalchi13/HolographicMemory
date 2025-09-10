from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .log_formatters import JSONFormatter


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rotating_json_handler(filepath: Path, level: int = logging.INFO, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Handler:
    ensure_dir(filepath.parent)
    handler = RotatingFileHandler(str(filepath), maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    return handler

