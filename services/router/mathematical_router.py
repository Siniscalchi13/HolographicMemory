"""
Mathematical Router

Combines threshold math and security guard to produce routing decisions.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from services.math_core import DimensionOptimizer, ThresholdCalculator
import os
from services.vault import SecurityGuard


LAYER_NAMES = [
    "identity",
    "knowledge",
    "experience",
    "preference",
    "context",
    "wisdom",
    "vault",
]


class MathematicalRouter:
    def __init__(self) -> None:
        self.optimizer = DimensionOptimizer()
        self.threshold = ThresholdCalculator()
        self.guard = SecurityGuard()

    def route_content(self, content: bytes, metadata: Dict) -> Dict:
        """Return routing decision.

        Output example:
        {
          "vault": bool,
          "format": "micro"|"v4"|"microK8",
          "layers": [(name, weight), ...],  # absent for vault
          "K": int,  # Top-K for v4 path
        }
        """
        # 1) Vault detection
        if self.guard.detect_secrets(content):
            return {
                "vault": True,
                "format": "micro",
                "layers": [("vault", 1.0)],
                "K": 0,
            }

        # 2) Size threshold decision
        size = len(content)
        micro_thr = int(os.getenv("HOLO_MICRO_THRESHOLD", "256") or 256)
        microk8_max = int(os.getenv("HOLO_MICRO_K8_MAX", "1024") or 1024)
        if size <= micro_thr:
            fmt = "micro"
        elif size <= microk8_max:
            fmt = "microK8"
        else:
            fmt = "v4"

        # 3) Layer routing (up to 2) if not vault
        layers = self._route_layers(metadata.get("filename", ""), metadata.get("content_type", ""), content, max_layers=2)
        if not layers:
            layers = [("knowledge", 1.0)]

        # 4) Top-K policy
        K = 0 if fmt == "micro" else (8 if fmt == "microK8" else int(os.getenv("HOLO_TOPK", "32") or 32))

        return {
            "vault": False,
            "format": fmt,
            "layers": layers,
            "K": K,
        }

    # ----------------- helpers -----------------
    @staticmethod
    def _ext(path: str) -> str:
        i = path.rfind('.')
        return path[i + 1:].lower() if i >= 0 else ''

    def _route_layers(self, filename: str, content_type: str, raw: bytes, max_layers: int = 2) -> List[Tuple[str, float]]:
        text = ""
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        scores: Dict[str, float] = {k: 0.0 for k in LAYER_NAMES}
        # priors
        scores["knowledge"] += 0.6
        scores["context"] += 0.2
        # extension/content type
        ext = self._ext(filename)
        if ext in ("pdf", "txt", "md", "doc", "docx", "rtf", "tex"):
            scores["knowledge"] += 0.4
        if ext in ("log", "csv", "tsv"):
            scores["experience"] += 0.4
        if ext in ("ini", "cfg", "conf", "yaml", "yml", "toml", "json"):
            scores["preference"] += 0.6
        if ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff"):
            scores["context"] += 0.3
        if content_type.startswith("text/"):
            scores["knowledge"] += 0.2
        if content_type.startswith("application/pdf"):
            scores["knowledge"] += 0.2
        # features
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or ""):
            scores["identity"] += 0.8
        if re.search(r"\b\w+ed\b", text or ""):
            scores["experience"] += 0.6
        if len(raw) < 2048:
            scores["preference"] += 0.2
        if any(t in (text or "").lower() for t in ("verified", "checksum", "signature", "ground truth", "validated")):
            scores["wisdom"] += 0.7
        # select top-k
        items = [(k, v) for k, v in scores.items() if v > 0 and k != "vault"]
        items.sort(key=lambda kv: kv[1], reverse=True)
        top = items[:max_layers] if items else [("knowledge", 1.0)]
        s = sum(w for _, w in top) or 1.0
        return [(k, w / s) for k, w in top]
