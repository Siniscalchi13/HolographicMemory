from __future__ import annotations

from typing import List, Dict

import pytest

from services.aiucp.verbum_field_engine.selection import (
    TASKS,
    intent_posterior,
    feature_vector,
)


@pytest.mark.parametrize(
    "fixture_name",
    ["messages_chat", "messages_code", "messages_reasoning", "messages_embeddings"],
)
def test_intent_posterior_shapes(request, fixture_name: str):
    messages: List[Dict[str, str]] = request.getfixturevalue(fixture_name)
    pi = intent_posterior(messages)
    assert isinstance(pi, list)
    assert len(pi) == len(TASKS)
    s = sum(pi)
    assert abs(s - 1.0) < 1e-6
    assert all(p >= 0.0 for p in pi)


def test_feature_vector_stability(messages_code):
    x1 = feature_vector(messages_code)
    x2 = feature_vector(messages_code)
    assert len(x1) == len(x2) >= 4
    # Deterministic for same input
    assert x1 == x2


def test_edge_cases_for_intents():
    empty: List[Dict[str, str]] = []
    pi = intent_posterior(empty)
    assert len(pi) == len(TASKS)
    assert abs(sum(pi) - 1.0) < 1e-6
    fv = feature_vector([{ "role": "user", "content": "" }])
    assert isinstance(fv, list) and len(fv) >= 4

