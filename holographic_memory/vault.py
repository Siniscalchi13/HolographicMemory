"""
Public SecureVault API

Re-exports the information-theoretic vault guard utilities.
"""
from __future__ import annotations

try:
    from services.vault import SecurityGuard as _Guard  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("services.vault not available in distribution") from exc


class SecureVault(_Guard):
    pass

