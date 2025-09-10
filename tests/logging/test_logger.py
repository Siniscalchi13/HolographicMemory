from __future__ import annotations

import json
import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .log_handlers import rotating_json_handler, ensure_dir


@dataclass
class EnterpriseTestLogger:
    """Enterprise-grade test logging system with structured output."""

    test_name: str
    log_dir: Path

    def __post_init__(self) -> None:
        ensure_dir(self.log_dir)
        self._lock = threading.Lock()
        self.setup_loggers()

    def setup_loggers(self) -> None:
        self.test_logger = self._create_logger(
            "test_execution", self.log_dir / "test_execution.log", level=logging.INFO
        )
        self.perf_logger = self._create_logger(
            "performance", self.log_dir / "performance.log", level=logging.INFO
        )
        self.security_logger = self._create_logger(
            "security", self.log_dir / "security.log", level=logging.WARNING
        )
        self.audit_logger = self._create_logger(
            "audit", self.log_dir / "audit.log", level=logging.INFO
        )
        self.compliance_logger = self._create_logger(
            "compliance", self.log_dir / "compliance.log", level=logging.INFO
        )

    def _create_logger(self, name: str, log_file: Path, level: int) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Reset handlers to avoid duplication across sessions
        for h in list(logger.handlers):
            logger.removeHandler(h)
        file_h = rotating_json_handler(log_file, level=level)
        console_h = logging.StreamHandler(sys.stdout)
        console_h.setLevel(level)
        # Reuse same JSON formatter for console
        console_h.setFormatter(file_h.formatter)  # type: ignore[attr-defined]
        logger.addHandler(file_h)
        logger.addHandler(console_h)
        return logger

    @contextmanager
    def test_step(self, step_name: str, details: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        start_time = datetime.utcnow()
        step_id = f"{self.test_name}_{step_name}_{start_time.timestamp()}"
        self.log_test_step_start(step_name, step_id, details or {})
        try:
            yield step_id
            self.log_test_step_success(step_name, step_id, start_time)
        except Exception as e:  # pragma: no cover - logging path
            self.log_test_step_failure(step_name, step_id, start_time, str(e))
            raise

    # --------------- Logging primitives ---------------
    def log_test_step_start(self, step: str, step_id: str, details: Dict[str, Any]) -> None:
        self.test_logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": self.test_name,
                    "step_id": step_id,
                    "step": step,
                    "status": "started",
                    "details": details,
                }
            )
        )

    def log_test_step_success(self, step: str, step_id: str, start_time: datetime) -> None:
        duration = (datetime.utcnow() - start_time).total_seconds()
        self.test_logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": self.test_name,
                    "step_id": step_id,
                    "step": step,
                    "status": "success",
                    "duration_seconds": duration,
                }
            )
        )

    def log_test_step_failure(self, step: str, step_id: str, start_time: datetime, error: str) -> None:
        duration = (datetime.utcnow() - start_time).total_seconds()
        self.test_logger.error(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": self.test_name,
                    "step_id": step_id,
                    "step": step,
                    "status": "failure",
                    "duration_seconds": duration,
                    "error": error,
                }
            )
        )

    def log_performance_metric(self, metric: str, value: float, unit: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.perf_logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": self.test_name,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "metadata": metadata or {},
                }
            )
        )

    def log_security_event(self, event: str, severity: str, details: Dict[str, Any]) -> None:
        self.security_logger.warning(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": self.test_name,
                    "event": event,
                    "severity": severity,
                    "details": details,
                }
            )
        )

    def log_audit_event(self, action: str, user: str, details: Dict[str, Any]) -> None:
        self.audit_logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": self.test_name,
                    "action": action,
                    "user": user,
                    "details": details,
                }
            )
        )

    def log_compliance_event(self, regulation: str, requirement: str, status: str, details: Dict[str, Any]) -> None:
        self.compliance_logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_name": self.test_name,
                    "regulation": regulation,
                    "requirement": requirement,
                    "status": status,
                    "details": details,
                }
            )
        )

