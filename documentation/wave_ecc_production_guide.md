# Wave ECC Production Guide (Phase 3.5)

This guide covers performance benchmarking, stress testing, configuration optimization, and monitoring for the Wave ECC subsystem.

## Overview
- GPU-only: Wave ECC is implemented in the native GPU binding (`holographic_gpu`). CPU fallbacks are not used in production.
- API surface:
  - `wave_ecc_encode(data: bytes, redundancy_level: int, seed_base: int) -> bytes`
  - `wave_ecc_decode(data: bytes, parity: bytes, redundancy_level: int, seed_base: int) -> tuple[bytes, int]`
- Integration: Containers store per-chunk parity (scheme=2). Adapter `verify_and_correct_rs()` calls Wave ECC, then parity revalidates.

## Benchmarking
Use the provided script to establish baselines and track regressions:

Command:
- `PYTHONPATH=build_holo python scripts/benchmark_wave_ecc_performance.py`

What it does:
- Sweeps sizes: 1KB, 10KB, 100KB, 1MB, 10MB (configurable)
- Tests redundancy levels: 1, 3, 5, 7 (configurable)
- Measures encode/decode latency (clean + light corruption)
- Tracks process RSS/CPU (via psutil)
- Saves report to `documentation/benchmarks/wave_ecc_performance_latest.json`

## Stress Testing
Long-running, large logical-size streaming tests with corruption and concurrency:

Command examples:
- `PYTHONPATH=build_holo python scripts/stress_test_wave_ecc.py --sizes 100MB 500MB --duration 600 --workers 2`
- For 24h burn-in: `--duration 86400`

What it does:
- Streams chunks to avoid high memory usage
- Injects configurable random byte flips per chunk
- Performs parity recheck post-decode to record success/failure
- Writes JSONL telemetry to `logs/wave_ecc_stress.jsonl`

## Configuration Optimization
Explore redundancy levels vs. error profiles and data types to derive presets:

Command:
- `PYTHONPATH=build_holo python scripts/optimize_wave_ecc_config.py`

What it does:
- Data types: text, binary, compressed, random
- Sizes: default 100KB (configurable)
- Redundancy sweep: k=1..10
- Error profiles: light (1e-4), moderate (5e-4), heavy (1e-3)
- Outputs recommendations and presets to `documentation/benchmarks/wave_ecc_config_recommendations.json`

## Monitoring & Observability
### Stats API
- `HoloFS.stats()` returns a `wave_ecc_metrics` dict with counts, success-rate, average latencies, and parity overhead.
- GPU info exposed under `stats()["gpu"]`.

### Telemetry
- Enable JSONL telemetry by setting `HOLO_WAVE_ECC_TELEMETRY=1` in the environment.
- Output file: `logs/wave_ecc_metrics.jsonl`
- Records per-encode/decode events with latency and process RSS/CPU.

### Dashboard
- Open `dashboard/wave_ecc_monitor.html` in a browser.
- Options:
  - Click "Connect API" to poll `/api/status` and `/api/metrics` if the Flask dashboard is running (`python dashboard/app.py`).
  - Or use the file input to load `logs/wave_ecc_metrics.jsonl` offline.

### Alerting (suggested thresholds)
- Decode success-rate < 99.5% over 5-minute window: warn
- Avg decode latency > 20 ms for 100KB payloads: warn
- Parity overhead > 0.6 for R<=5: warn

## Presets (Initial)
Derived from the optimizer’s default run. Adjust to your environment.

- Balanced: `redundancy=3`
- High Reliability: `redundancy=5`
- Low Overhead: `redundancy=2`

Store these as container defaults per workload profile and update via periodic re-optimization.

## Troubleshooting
- "Wave ECC binding not available": ensure `build_holo` is built and `PYTHONPATH=build_holo` is set.
- Decode failures: check that parity recheck is not being bypassed; validate seed base (`ecc_r`) is consistent.
- No GPU detected: confirm Metal backend availability and that `HolographicGPU.initialize()` succeeds.

## Deployment Checklist
- [ ] Build native GPU extension (`make native` or `cmake --build build_holo -j 4`)
- [ ] Validate Wave ECC tests pass (`test_wave_ecc.py` and core tests)
- [ ] Run benchmarks and save report
- [ ] Run stress test (≥ 1 hour, then extended 24h window during staging)
- [ ] Generate config recommendations and set presets
- [ ] Enable telemetry in production (`HOLO_WAVE_ECC_TELEMETRY=1`)
- [ ] Integrate dashboard monitoring and alert thresholds

