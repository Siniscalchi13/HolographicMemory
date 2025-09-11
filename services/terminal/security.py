from __future__ import annotations

import os
from typing import Optional


def validate_token(token: Optional[str]) -> bool:
    """Basic token validation. Anonymous access disabled by default."""
    expected = os.environ.get("TERMINAL_AUTH_TOKEN")
    if expected:
        return (token or "") == expected
    # If no expected token configured, deny by default
    return False
