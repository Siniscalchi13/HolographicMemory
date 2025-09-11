from __future__ import annotations

import secrets
from typing import Dict, Optional


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, str] = {}

    def create(self, username: str) -> str:
        token = secrets.token_urlsafe(24)
        self.sessions[token] = username
        return token

    def get_user(self, token: str) -> Optional[str]:
        return self.sessions.get(token)

    def destroy(self, token: str) -> None:
        self.sessions.pop(token, None)

