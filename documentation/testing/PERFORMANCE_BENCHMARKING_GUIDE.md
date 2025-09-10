# Performance Benchmarking Guide

## Microbenchmarks

- Use `pytest-benchmark` under `tests/performance/`.
- Keep deterministic; avoid network or GPU in CI.

## Load/Stress (Out of band)

- For 1000+ concurrent users, use Locust or k6 against the API service.
- Define targets in `tests/config/performance_config.yaml`.

## Reporting

- Commit benchmark JSON summaries to CI artifacts.
- Track regressions across builds with trend charts.

