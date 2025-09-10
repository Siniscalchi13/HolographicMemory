# Security Testing Procedures

## Static Analysis

- Run: `pytest -m security -k vulnerability_scanning`
- Tools: `bandit`, `pip-audit` (non-blocking in CI)

## Data Protection

- Vault routing is validated by unit and integration tests.
- Never store test data with real secrets; use generators under `tests/utils/`.

## Compliance

- Extend `tests/compliance/` with audit, retention, and privacy checks as policies evolve.

