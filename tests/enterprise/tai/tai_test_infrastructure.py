from __future__ import annotations

import os
import sys
from pathlib import Path


def test_imports_and_services_available():
    # Ensure core packages import
    import importlib

    vfe_sel = importlib.import_module("services.aiucp.verbum_field_engine.selection")
    vfe_par = importlib.import_module("services.aiucp.verbum_field_engine.pareto")
    nsga2 = importlib.import_module("services.aiucp.verbum_field_engine.nsga2_optimizer")
    qcore = importlib.import_module("services.aiucp.quantum_core.analytics")
    hmc_calc = importlib.import_module("services.aiucp.holographic_memory.phase_retrieval")
    hmc_mem = importlib.import_module("services.aiucp.holographic_memory.pure_python_memory")
    aioc = importlib.import_module("services.aiucp.quantum_core.advanced_lyapunov")
    ppc = importlib.import_module("services.aiucp.quantum_core.enhanced_privacy")

    assert hasattr(vfe_sel, "selection_scores")
    assert hasattr(vfe_par, "ParetoOptimizationCalculus")
    assert hasattr(nsga2, "nsga2")
    assert hasattr(qcore, "compute_quantum_analytics")
    assert hasattr(hmc_calc, "PhaseRetrievalCalculus")
    assert hasattr(hmc_mem, "HolographicMemory")
    assert hasattr(aioc, "quadratic_stability_region")
    assert hasattr(ppc, "gaussian_pld")


def test_infrastructure_artifacts_present():
    # Helm charts must exist for services
    charts = [
        Path("charts/vfe/Chart.yaml"),
        Path("charts/hmc/Chart.yaml"),
        Path("charts/aiocp/Chart.yaml"),
        Path("charts/observability/Chart.yaml"),
    ]
    for p in charts:
        assert p.is_file(), f"missing chart: {p}"
    # Monitoring dashboards present
    for dash in [
        Path("docs/monitoring/grafana-vfe-dashboard.json"),
        Path("docs/monitoring/grafana-hmc-dashboard.json"),
        Path("docs/monitoring/grafana-qec-dashboard.json"),
    ]:
        assert dash.is_file(), f"missing dashboard: {dash}"

