from __future__ import annotations

from typing import Any, Dict, List

from services.aiucp.verbum_field_engine.selection import selection_scores


def _models() -> List[Dict[str, Any]]:
    return [
        {"name": "apple-small", "backend": "apple", "ctx": 4096, "A": [[0.2, 0.3, 0.1, 0.0, 0.5]]},
        {"name": "llama-fast", "backend": "llama.cpp", "ctx": 8192, "A": [[0.1, 0.6, 0.2, 0.0, 0.3]]},
        {"name": "mlx-balanced", "backend": "mlx", "ctx": 32768, "A": [[0.2, 0.3, 0.4, 0.0, 0.4]]},
        {"name": "remote-accurate", "backend": "openai", "ctx": 16384, "A": [[0.3, 0.6, 0.4, 0.0, 0.2]]},
    ]


def _messages() -> List[Dict[str, str]]:
    return [{"role": "user", "content": "Write Python code to compute eigenvalues."}]


def test_scalar_selection():
    sel, scores = selection_scores(_models(), _messages(), max_tokens=128)
    assert sel in scores


def test_pareto_mode():
    params = {"pareto": 1, "pareto_weights": [0.6, 0.3, 0.1]}
    sel, scores = selection_scores(_models(), _messages(), max_tokens=128, params=params)
    assert sel in scores


def test_nsga2_mode():
    params = {"pareto": 1, "nsga2": True}
    sel, scores = selection_scores(_models(), _messages(), max_tokens=128, params=params)
    assert sel in scores


def test_ab_bonus():
    models = [
        {"name": "A", "backend": "llama.cpp", "ctx": 4096, "A": [[0.2, 0.2, 0.2, 0.0, 0.6]]},
        {"name": "B", "backend": "llama.cpp", "ctx": 4096, "A": [[0.2, 0.2, 0.2, 0.0, 0.6]]},
    ]

    def bonus(name: str, x: List[float]) -> float:
        return 0.05 if name == "B" else 0.0

    sel, _ = selection_scores(models, _messages(), max_tokens=128, bonus_fn=bonus)
    assert sel == "B"

