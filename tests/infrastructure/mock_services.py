from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class MockMemory:
    state: Dict[str, bytes] = field(default_factory=dict)
    grid_size: int = 64
    use_gpu: bool = False
    backend = object()

    def store_bytes(self, doc_id: str, content: bytes) -> Dict[str, Any]:
        self.state[str(doc_id)] = bytes(content)
        return {"ok": True, "encoded_data": content[:16]}

    def retrieve_bytes(self, doc_id: str) -> bytes:
        return self.state.get(str(doc_id), b"")


class MockRouter:
    def route_content(self, content: bytes, metadata: Dict) -> Dict:
        # simple heuristic: vault if contains 'secret'
        if b"secret" in content:
            return {"vault": True, "format": "micro", "layers": [("vault", 1.0)], "K": 0}
        fmt = "micro" if len(content) <= 256 else ("microK8" if len(content) <= 1024 else "v4")
        return {"vault": False, "format": fmt, "layers": [("knowledge", 1.0)], "K": 0 if fmt == "micro" else (8 if fmt == "microK8" else 32)}


class MockVault:
    def detect_secrets(self, content: bytes) -> bool:
        return b"secret" in content or b"API_KEY" in content

    def generate_vault_id(self) -> str:
        return "deadbeefdeadbeef"

