from __future__ import annotations

import os
from typing import Optional


def validate_token(token: Optional[str]) -> bool:
    """Basic token validation with development mode fallback."""
    expected = os.environ.get("TERMINAL_AUTH_TOKEN")
    if expected:
        return (token or "") == expected
    
    # Development mode: allow access if no token is configured
    # This prevents lockout during development
    dev_mode = os.environ.get("TERMINAL_DEV_MODE", "true").lower() == "true"
    if dev_mode and not expected:
        return True
    
    # If no expected token configured, deny by default
    return False
