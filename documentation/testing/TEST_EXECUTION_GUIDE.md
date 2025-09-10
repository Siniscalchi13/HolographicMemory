# Enterprise Test Execution Guide

## Quick Start

- Install dev deps: `pip install -e services/holographic-memory/core .[dev,enterprise]`
- Run core suites: `pytest -m "unit or integration or contract"`
- Run e2e: `pytest -m e2e`
- Generate HTML report: `pytest --html=tests/reports/pytest_report.html --self-contained-html`

## Markers

- `unit` — fast, isolated tests
- `integration` — cross-service interactions and API Client
- `contract` — OpenAPI/schema
- `e2e` — end-to-end flows
- `performance` — microbenchmarks, load (non-blocking in CI)
- `security` — bandit/pip-audit (non-blocking in CI)
- `chaos` — failure injection (opt-in)

## Coverage

- HTML coverage is emitted to `tests/reports/coverage/`
- Thresholds enforced at gate level in CI

## Environment

- Tests isolate `HLOG_DATA_DIR`/`HOLOGRAPHIC_DATA_DIR` to a temp folder
- GPU tests are optional and skipped if backends are unavailable

