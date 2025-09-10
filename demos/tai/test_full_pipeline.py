from __future__ import annotations

from typing import Any, Dict, List

import pytest

from services.aiucp.verbum_field_engine.selection import selection_scores


def _execute_stub(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate LEF execution by validating and echoing the plan."""
    assert cmd["command"] == "run_model"
    # In real LEF, this would dispatch to local backend; here we just return ack
    return {
        "ok": True,
        "model": cmd["model"],
        "backend": cmd["backend"],
        "tokens": cmd["params"].get("max_tokens", 0),
        "input_size": sum(len(m.get("content", "")) for m in cmd["input"]),
    }


def test_pipeline_simple_chat(sample_models, messages_chat, lef_synthesizer):
    sel, scores = selection_scores(sample_models, messages_chat, max_tokens=128)
    cmd = lef_synthesizer(sel, sample_models, messages_chat, max_tokens=128)
    result = _execute_stub(cmd)
    assert result["ok"] and result["model"] == sel


def test_pipeline_complex_code(sample_models, messages_code, lef_synthesizer):
    sel, scores = selection_scores(sample_models, messages_code, max_tokens=256)
    cmd = lef_synthesizer(sel, sample_models, messages_code, max_tokens=256, extra={"temperature": 0.2})
    result = _execute_stub(cmd)
    assert result["ok"] and result["tokens"] == 256


def test_pipeline_multi_objective(sample_models, messages_chat, lef_synthesizer):
    params = {"pareto": 1, "pareto_weights": [0.6, 0.3, 0.1]}
    sel, scores = selection_scores(sample_models, messages_chat, max_tokens=96, params=params)
    cmd = lef_synthesizer(sel, sample_models, messages_chat, max_tokens=96, extra={"strategy": "pareto"})
    result = _execute_stub(cmd)
    assert result["ok"] and cmd["params"]["strategy"] == "pareto"


def test_pipeline_privacy_intent(sample_models, lef_synthesizer):
    messages_privacy = [
        {"role": "user", "content": "Analyze this data while maintaining differential privacy."}
    ]
    # Selection should still succeed; attach privacy flags to LEF params
    sel, scores = selection_scores(sample_models, messages_privacy, max_tokens=64)
    cmd = lef_synthesizer(sel, sample_models, messages_privacy, max_tokens=64, extra={"differential_privacy": True, "epsilon": 1.0})
    result = _execute_stub(cmd)
    assert result["ok"] and cmd["params"]["differential_privacy"] is True

