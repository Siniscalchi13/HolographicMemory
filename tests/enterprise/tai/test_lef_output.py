from __future__ import annotations

from typing import Any, Dict

from services.aiucp.verbum_field_engine.selection import selection_scores


def _validate_lef_schema(cmd: Dict[str, Any]) -> None:
    assert cmd.get("command") == "run_model"
    assert isinstance(cmd.get("model"), str) and cmd["model"]
    assert isinstance(cmd.get("backend"), str) and cmd["backend"]
    assert isinstance(cmd.get("params"), dict)
    assert isinstance(cmd.get("input"), list)
    assert isinstance(cmd["params"].get("max_tokens"), int)


def test_output_format_and_params(sample_models, messages_embeddings, lef_synthesizer):
    sel, scores = selection_scores(sample_models, messages_embeddings, max_tokens=128)
    cmd = lef_synthesizer(sel, sample_models, messages_embeddings, max_tokens=128, extra={"temperature": 0.1, "top_p": 0.95})
    _validate_lef_schema(cmd)
    # Parameter propagation
    assert cmd["params"]["temperature"] == 0.1
    assert cmd["params"]["top_p"] == 0.95
    # Completeness
    assert cmd["model"] in scores

