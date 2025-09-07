# HolographicMemory Enterprise Benchmark Suite

This suite provides Fortune 500-grade benchmarking and validation for HolographicMemory.

Highlights:
- Holographic chunked storage performance (256KB chunks)
- Bit-perfect reconstruction validation across file types
- Search performance vs ripgrep (filesystem) and SQLite (database)
- Compression and storage efficiency comparisons
- Real-world workflow and concurrent operations benchmarks
- Competitor benchmark placeholders (OneDrive/Dropbox/Google Drive) gated by env
- Statistical rigor with confidence intervals

Structure:
- `benchmarks/enterprise/benchmark_service/` – Ported TAI benchmark orchestrator
- `benchmarks/enterprise/holographic_benchmarks.py` – Holographic-specific tests
- `benchmarks/enterprise/statistics.py` – CI and validation helpers
- `tests/enterprise/conftest.py` – Skips TAI reference tests by default

Usage:
- Run all holographic benchmarks: `pytest -q benchmarks/enterprise`
- Focus on subset: `scripts/run_enterprise_benchmarks.py --suite holographic`
- Enable larger datasets: set `HM_BENCH_MEDIUM=1` or `HM_BENCH_LARGE=1`
- Enable competitor tests: set `HM_ENABLE_COMPETITOR=1` (requires CLI setup)

Reports:
- JSON/HDF5 via TAI reporting service (optional)
- Markdown summary: `reports/benchmarks/enterprise_report.md`

