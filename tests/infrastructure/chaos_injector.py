from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator


@contextmanager
def inject_exception(obj, name: str, exc: Exception) -> Iterator[None]:
    original = getattr(obj, name)

    def _boom(*_a, **_k):  # noqa: ANN001, ANN002, ANN003
        raise exc

    setattr(obj, name, _boom)
    try:
        yield
    finally:
        setattr(obj, name, original)

