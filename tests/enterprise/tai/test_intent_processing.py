from __future__ import annotations

import math
from typing import List, Dict

import numpy as np

from services.aiucp.verbum_field_engine.selection import (
    intent_posterior,
    capability_logits,
    softmax,
    feature_vector,
    kernel_alignment,
)


def test_posterior_and_feature_vector(messages_reasoning):
    pi = intent_posterior(messages_reasoning)
    x = feature_vector(messages_reasoning)
    assert len(pi) >= 3 and abs(sum(pi) - 1.0) < 1e-6
    assert isinstance(x, list) and len(x) >= 4


def test_capability_logits_and_softmax(messages_code):
    x = feature_vector(messages_code)
    # 4 tasks x d features (truncate/align if oversized)
    A = [
        [0.1, 0.0, 0.0, 0.0, 0.2],
        [0.0, 0.6, 0.2, 0.0, 0.1],
        [0.0, 0.2, 0.5, 0.0, 0.1],
        [0.0, 0.0, 0.0, 0.5, 0.2],
    ]
    logits = capability_logits(A, x)
    sm = softmax(logits)
    assert len(sm) == len(A)
    assert min(sm) >= 0.0 and abs(sum(sm) - 1.0) < 1e-9


def test_kernel_alignment(messages_chat):
    x = feature_vector(messages_chat)
    mu = [0.1, 0.0, 0.0, 0.0, 1.0]
    k = kernel_alignment(x, mu, sigma=1.0)
    assert 0.0 <= k <= 1.0

