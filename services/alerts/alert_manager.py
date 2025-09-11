from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Alert:
    level: str
    message: str


class AlertManager:
    def __init__(self):
        self.alerts: List[Alert] = []

    def emit(self, level: str, message: str) -> None:
        self.alerts.append(Alert(level, message))

    def recent(self, limit: int = 50) -> List[Alert]:
        return self.alerts[-limit:]

