from __future__ import annotations

import os
from typing import Iterator


def random_bytes(size: int) -> bytes:
    return os.urandom(max(0, int(size)))


def large_bytes(total_size: int, chunk: int = 1 << 20) -> Iterator[bytes]:
    """Yield random chunks up to total_size bytes for streaming tests."""
    remaining = max(0, int(total_size))
    chunk = max(1, int(chunk))
    while remaining > 0:
        n = min(chunk, remaining)
        yield os.urandom(n)
        remaining -= n

