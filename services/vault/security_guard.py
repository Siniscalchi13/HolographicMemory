"""
Vault Security Guard

Implements secret detection and Vault doc_id generation per
documentation/specifications/VAULT_SECURITY_SPECIFICATION.md.
"""
from __future__ import annotations

import os
import re
import secrets
from typing import Pattern


class SecurityGuard:
    """Secret detection and Vault routing primitives.

    detect_secrets uses pattern and entropy-based heuristics to identify
    sensitive payloads (API keys, JWTs, .env tokens, passwords).
    generate_vault_id returns a random opaque identifier independent of content.
    """

    # Common secret/token indicators and JWT pattern
    _PATTERNS: list[Pattern[str]] = [
        re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token|bearer)\b"),
        re.compile(r"(?i)\baws[_-]?secret[_-]?access[_-]?key\b"),
        re.compile(r"(?i)\bghp_[A-Za-z0-9]{20,}\b"),  # GitHub PAT style
        re.compile(r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}"),  # JWT
    ]

    def __init__(self, entropy_threshold: float = 4.0) -> None:
        self.entropy_threshold = float(entropy_threshold)

    def detect_secrets(self, content: bytes) -> bool:
        """Return True if content appears to contain secrets.

        Heuristics:
          - keyword patterns
          - JWT-like triple base64url segments
          - high-entropy tokens (Shannon entropy per byte > threshold for >= 32B windows)
        """
        if not content:
            return False
        text = None
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = None
        if text:
            for pat in self._PATTERNS:
                if pat.search(text):
                    return True
        # entropy check on raw bytes (windowed)
        if len(content) >= 32 and self._window_high_entropy(content, 32):
            return True
        return False

    def generate_vault_id(self) -> str:
        """Generate a random 128-bit identifier encoded as hex (content-independent)."""
        return secrets.token_hex(16)

    # ----------------- helpers -----------------
    def _window_high_entropy(self, data: bytes, w: int) -> bool:
        for i in range(0, len(data) - w + 1, max(8, w // 2)):
            if self._shannon_entropy(data[i : i + w]) >= self.entropy_threshold:
                return True
        return False

    @staticmethod
    def _shannon_entropy(buf: bytes) -> float:
        import math
        if not buf:
            return 0.0
        counts = {}
        for b in buf:
            counts[b] = counts.get(b, 0) + 1
        H = 0.0
        n = float(len(buf))
        for c in counts.values():
            p = c / n
            H -= p * math.log2(p)
        # normalize to per-byte entropy (0..8)
        return H

