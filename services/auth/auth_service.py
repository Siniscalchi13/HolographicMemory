from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    username: str
    role: str = "user"


class AuthService:
    def authenticate(self, username: str, password: str) -> Optional[User]:
        # Dev-only stub. Replace with real auth.
        if username and password:
            return User(username=username)
        return None

