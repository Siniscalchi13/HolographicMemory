from __future__ import annotations

import re
from typing import Dict, List, Tuple


LAYER_NAMES = [
    "identity",
    "knowledge",
    "experience",
    "preference",
    "context",
    "wisdom",
]


def _ext(path: str) -> str:
    i = path.rfind('.')
    return path[i+1:].lower() if i >= 0 else ''


def _is_config_like(name: str, content_type: str) -> bool:
    ext = _ext(name)
    if ext in ("ini", "cfg", "conf", "yaml", "yml", "toml", "json"):
        return True
    if content_type.startswith("application/json"):
        return True
    return False


def _has_past_tense(text: str) -> bool:
    # crude heuristic: words ending in 'ed' and presence of dates
    return bool(re.search(r"\b\w+ed\b", text)) or bool(re.search(r"\b20\d{2}\b", text))


def _has_identity(text: str) -> bool:
    # emails/usernames/handles heuristics
    return bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)) or "username" in text.lower()


def _has_wisdom_markers(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("verified", "checksum", "signature", "ground truth", "validated"))


def route_layers(filename: str, content_type: str, raw: bytes, max_layers: int = 2) -> List[Tuple[str, float]]:
    """Return up to max_layers (layer, weight) pairs.

    Heuristics prefer knowledge & context by default, with boosts for identity,
    experience, preference, and wisdom based on filename/content signals.
    """
    size = len(raw)
    text = ""
    try:
        # attempt lightweight text decode
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = ""

    scores: Dict[str, float] = {k: 0.0 for k in LAYER_NAMES}

    # Base priors
    scores["knowledge"] += 0.6
    scores["context"] += 0.2

    # Extension-based routing
    ext = _ext(filename)
    if ext in ("pdf", "txt", "md", "doc", "docx", "rtf", "tex"):
        scores["knowledge"] += 0.4
    if ext in ("log", "csv", "tsv"):
        scores["experience"] += 0.4
    if ext in ("ini", "cfg", "conf", "yaml", "yml", "toml", "json"):
        scores["preference"] += 0.6
    if ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff"):
        scores["context"] += 0.3

    # Content type hints
    if content_type.startswith("text/"):
        scores["knowledge"] += 0.2
    if content_type.startswith("application/pdf"):
        scores["knowledge"] += 0.2

    # Content heuristics
    if _has_identity(text):
        scores["identity"] += 0.8
    if _has_past_tense(text):
        scores["experience"] += 0.6
    if _is_config_like(filename, content_type):
        scores["preference"] += 0.4
    if size < 2048:
        scores["preference"] += 0.2
    if _has_wisdom_markers(text):
        scores["wisdom"] += 0.7

    # Normalize positive scores and select top-k
    items = [(k, v) for k, v in scores.items() if v > 0.0]
    if not items:
        return [("knowledge", 1.0)]
    items.sort(key=lambda kv: kv[1], reverse=True)
    top = items[:max_layers]
    s = sum(w for _, w in top)
    if s <= 0:
        return [("knowledge", 1.0)]
    return [(k, w / s) for k, w in top]

