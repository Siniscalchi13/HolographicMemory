from __future__ import annotations

from typing import Dict, Set


class Permissions:
    def __init__(self):
        self.roles: Dict[str, Set[str]] = {
            "admin": {"terminal", "metrics", "manage_services"},
            "user": {"terminal", "metrics"},
        }

    def allowed(self, role: str, perm: str) -> bool:
        return perm in self.roles.get(role, set())

