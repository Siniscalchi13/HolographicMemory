from __future__ import annotations

from typing import Iterable


class NotificationService:
    def __init__(self):
        self.subscribers: list[callable] = []

    def subscribe(self, fn: callable) -> None:
        self.subscribers.append(fn)

    def notify(self, message: str) -> None:
        for fn in list(self.subscribers):
            try:
                fn(message)
            except Exception:
                pass

