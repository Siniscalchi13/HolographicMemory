from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List


def test_environment_and_imports():
    # Ensure alias module path works (underscore → hyphen directory)
    vfe_sel = importlib.import_module("services.aiucp.verbum_field_engine.selection")
    vfe_par = importlib.import_module("services.aiucp.verbum_field_engine.pareto")
    qc_mod = importlib.import_module("services.aiucp.quantum_core.analytics")
    # Smoke check of required symbols
    assert hasattr(vfe_sel, "selection_scores")
    assert hasattr(vfe_sel, "feature_vector")
    assert hasattr(vfe_par, "ParetoOptimizationCalculus")
    assert hasattr(qc_mod, "compute_quantum_analytics")


def test_mock_aios_and_basic_integration(sample_models, messages_chat):
    # Simulate AIOS intent → LQL selection → scores
    from services.aiucp.verbum_field_engine.selection import selection_scores

    sel_name, scores = selection_scores(sample_models, messages_chat, max_tokens=128)
    assert sel_name in {m["name"] for m in sample_models}
    assert isinstance(scores, dict) and len(scores) == len(sample_models)


def test_lef_output_synthesis(sample_models, messages_code, lef_synthesizer):
    from services.aiucp.verbum_field_engine.selection import selection_scores

    sel_name, _ = selection_scores(sample_models, messages_code, max_tokens=256)
    cmd = lef_synthesizer(sel_name, sample_models, messages_code, max_tokens=256, extra={"temperature": 0.2})
    # Minimal schema validation
    assert cmd["command"] == "run_model"
    assert cmd["model"] == sel_name
    assert cmd["backend"] in {m["backend"] for m in sample_models}
    assert "params" in cmd and isinstance(cmd["params"], dict)
    assert "input" in cmd and isinstance(cmd["input"], list)


def test_mathematical_verification_hook():
    # Validate a subset of analytics quickly to ensure math is wired up
    import numpy as np
    from services.aiucp.quantum_core.analytics import compute_quantum_analytics

    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 density matrix
    bell = np.zeros((4, 4), dtype=complex)
    bell[0, 0] = bell[3, 3] = 0.5
    bell[0, 3] = bell[3, 0] = 0.5
    H = np.diag([1.0, 0.0, 0.0, -1.0])
    out = compute_quantum_analytics(bell, dims=(2, 2), H=H)
    assert out["entropy_bits"] >= 0.0
    assert 0.8 <= out.get("concurrence", 0.0) <= 1.0
    assert abs(out.get("fidelity_self", 1.0) - 1.0) < 1e-12
    assert out.get("qfi_unitary", 0.0) >= 0.0
