from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Any


class RingBuffer:
    def __init__(self, maxlen: int = 1000):
        self.buf: Deque[Dict[str, Any]] = deque(maxlen=maxlen)

    def append(self, item: Dict[str, Any]) -> None:
        self.buf.append(item)

    def all(self):
        return list(self.buf)

