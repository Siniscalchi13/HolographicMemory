# Test Maintenance Procedures

- Keep unit tests deterministic (<1s each); avoid network and disk unless mocked.
- Prefer `pytest` fixtures over ad-hoc setup/teardown.
- Use markers consistently to scope CI jobs.
- Add new service tests under `tests/unit/<service>` and `tests/integration` as needed.
- Update `pyproject.toml` markers and coverage as services grow.

## Performance Baselines

- Store latency benchmarks live under `tests/performance/`.
- Update targets in `tests/config/performance_config.yaml` when hardware or SLAs change.

## Security

- Extend security tests with bandit rules and dependency audits as policies evolve.
- Never include real secrets or PII in fixtures or test data.

