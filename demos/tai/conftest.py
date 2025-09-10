from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure repository root on sys.path for service imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _stable_env(monkeypatch: pytest.MonkeyPatch):
    """Stabilize environment for deterministic tests."""
    # Prefer heuristic token counting to avoid tiktoken dependency in CI
    monkeypatch.setenv("VFE_DISABLE_TIKTOKEN", "1")
    # Default: disable pareto unless a test enables it explicitly
    monkeypatch.delenv("VFE_PARETO_ENABLE", raising=False)
    # Default: disable NSGA-II unless enabled
    monkeypatch.delenv("VFE_USE_NSGA2", raising=False)
    # Keep enhanced WF disabled by default (phase retrieval)
    monkeypatch.setenv("HMC_USE_ENHANCED_WF", "0")
    # Normalize mode for fallbacks
    monkeypatch.setenv("HMC_MODE", "mvp")
    yield


@pytest.fixture
def messages_chat() -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": "Help me understand quantum computing basics."},
    ]


@pytest.fixture
def messages_code() -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": "Write a Python function for quantum state preparation using numpy."},
        {"role": "user", "content": "Include def prepare_state(theta): and return a vector."},
    ]


@pytest.fixture
def messages_reasoning() -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": "Prove a simple lemma: therefore the theorem follows by induction."},
    ]


@pytest.fixture
def messages_embeddings() -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": "Compute sentence embeddings and return the vector as JSON."},
    ]


@pytest.fixture
def sample_models() -> List[Dict[str, Any]]:
    """Representative model registry sample with calibration and competence."""
    # 5-dim features expected by selection.feature_vector
    # Provide some variety across backends and contexts
    return [
        {
            "name": "apple-local-small",
            "backend": "apple",
            "ctx": 4096,
            "calibration": {"a": 0.7, "b": 1.0, "d": 0.0008},
            # single-row A used as direct competence linear model
            "A": [[0.2, 0.3, 0.1, 0.0, 0.5]],
            "mu": [0.2, 0.0, 0.0, 0.0, 1.0],
        },
        {
            "name": "llama-cpp-fast",
            "backend": "llama.cpp",
            "ctx": 8192,
            "calibration": {"a": 1.0, "b": 1.2, "d": 0.0010},
            # multi-row A → softmax over tasks
            "A": [
                [0.1, 0.0, 0.0, 0.0, 0.3],  # chat
                [0.0, 0.5, 0.1, 0.0, 0.2],  # code
                [0.0, 0.1, 0.4, 0.0, 0.1],  # reasoning
                [0.0, 0.0, 0.0, 0.5, 0.2],  # embeddings
            ],
            "mu": [0.1, 0.2, 0.2, 0.1, 1.0],
        },
        {
            "name": "mlx-balanced",
            "backend": "mlx",
            "ctx": 32768,
            "calibration": {"a": 1.1, "b": 1.0, "d": 0.0005},
            "A": [
                [0.3, 0.1, 0.1, 0.0, 0.3],
                [0.0, 0.3, 0.1, 0.0, 0.2],
                [0.0, 0.2, 0.4, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.3, 0.3],
            ],
            "mu": [0.1, 0.1, 0.2, 0.1, 1.0],
        },
        {
            "name": "remote-accurate",
            "backend": "openai",  # treated as remote (p_m = 0)
            "ctx": 16384,
            "calibration": {"a": 1.3, "b": 1.0, "d": 0.0005},
            "A": [
                [0.4, 0.1, 0.0, 0.0, 0.3],
                [0.0, 0.6, 0.2, 0.0, 0.2],
                [0.0, 0.1, 0.5, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.2, 0.2],
            ],
            "mu": [0.1, 0.2, 0.2, 0.05, 1.0],
        },
    ]


@pytest.fixture
def lef_synthesizer():
    """Synthesize a minimal LEF-compatible command from selection output.

    Schema:
    {
      "command": "run_model",
      "model": <name>,
      "backend": <backend>,
      "params": {"max_tokens": int, ...},
      "input": messages,
    }
    """

    def _make(selected_name: str, models: List[Dict[str, Any]], messages: List[Dict[str, str]], max_tokens: int = 256, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        extra = extra or {}
        mdl = next((m for m in models if m.get("name") == selected_name), None)
        assert mdl is not None, f"unknown model: {selected_name}"
        cmd = {
            "command": "run_model",
            "model": mdl["name"],
            "backend": mdl.get("backend", "unknown"),
            "params": {"max_tokens": int(max_tokens)} | extra,
            "input": messages,
        }
        # basic schema sanity
        assert isinstance(cmd["params"]["max_tokens"], int)
        return cmd

    return _make


@pytest.fixture
def record_report(tmp_path):
    """Provide a simple hook to record per-test artifacts into reports/enterprise."""
    out_dir = os.path.join("reports", "enterprise")
    os.makedirs(out_dir, exist_ok=True)

    def _write(name: str, payload: dict):
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    return _write
