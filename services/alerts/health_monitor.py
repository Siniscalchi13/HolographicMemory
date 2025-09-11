from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class HealthCheck:
    name: str
    check: Callable[[], bool]


class HealthMonitor:
    def __init__(self):
        self.checks: List[HealthCheck] = []

    def register(self, name: str, func: Callable[[], bool]) -> None:
        self.checks.append(HealthCheck(name, func))

    def run(self) -> Dict[str, bool]:
        return {hc.name: bool(hc.check()) for hc in self.checks}

