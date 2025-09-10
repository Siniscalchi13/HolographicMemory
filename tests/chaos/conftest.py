from __future__ import annotations

import contextlib
import random
from typing import Iterator

import pytest


@contextlib.contextmanager
def flaky_network(probability: float = 0.2) -> Iterator[None]:
    """Context manager to simulate intermittent failures.

    Users can wrap network calls; here we only provide the helper.
    """
    yield

