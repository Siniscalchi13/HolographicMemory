from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pytest

from services.aiucp.verbum_field_engine.selection import selection_scores


def test_scalar_selection_default(sample_models, messages_chat):
    sel, scores = selection_scores(sample_models, messages_chat, max_tokens=128)
    assert sel in {m["name"] for m in sample_models}
    assert len(scores) == len(sample_models)


def test_pareto_selection_mode(sample_models, messages_code, monkeypatch: pytest.MonkeyPatch):
    # Enable Pareto via params (prefer local over env for determinism)
    params = {"pareto": 1, "pareto_weights": [0.6, 0.3, 0.1]}
    sel, scores = selection_scores(sample_models, messages_code, max_tokens=256, params=params)
    assert sel in scores
    # Pareto should still return a valid model name
    assert sel in {m["name"] for m in sample_models}


def test_nsga2_evolutionary_mode(sample_models, messages_reasoning):
    # NSGA-II path is only triggered when Pareto is enabled and frontier_points exist
    params = {"pareto": 1, "nsga2": True}
    sel, scores = selection_scores(sample_models, messages_reasoning, max_tokens=192, params=params)
    assert sel in scores


def test_ab_testing_bonus_path(messages_chat):
    # Two identical models except name; bonus drives A/B choice
    models = [
        {
            "name": "model-A",
            "backend": "llama.cpp",
            "ctx": 4096,
            "calibration": {"a": 1.0, "b": 1.0, "d": 0.001},
            "A": [[0.2, 0.2, 0.2, 0.0, 0.6]],
        },
        {
            "name": "model-B",
            "backend": "llama.cpp",
            "ctx": 4096,
            "calibration": {"a": 1.0, "b": 1.0, "d": 0.001},
            "A": [[0.2, 0.2, 0.2, 0.0, 0.6]],
        },
    ]

    def bonus_fn(name: str, x: List[float]) -> float:
        # e.g., A/B group allocation favors model-B by a small constant
        return 0.05 if name.endswith("B") else 0.0

    sel, scores = selection_scores(models, messages_chat, max_tokens=128, bonus_fn=bonus_fn)
    assert sel == "model-B"

