from __future__ import annotations

import base64
import os
from typing import Tuple


def generate_test_token(payload: bytes = b"test") -> str:
    """Return a simple unsigned token for testing (not JWT)."""
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def random_secret(length: int = 32) -> bytes:
    return os.urandom(max(16, int(length)))

