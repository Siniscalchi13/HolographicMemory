# Reproducible Benchmarks (Enterprise)

This guide documents how to measure Holographic Memory performance in a way that is transparent, repeatable, and fair. It is intentionally conservative and “brutally honest” about what affects the numbers.

## Goals
- Report performance as ranges with error bars and conditions
- Separate doc_id recall (O(1) in N) from semantic search (O(n))
- Capture environment (CPU, OS, Python) and build details
- Produce machine-readable JSON reports + human summaries

## What You Measure
- Store throughput (ops/s) with looped batches
- doc_id recall latency vs corpus size (O(1) claim)
- Semantic search latency vs corpus size (expected O(n) unless ANN index used)

## How To Run

```bash
# Ensure native extensions are built (Release)
make cpp

# Optional: enable a performance profile
# macOS: ensure high-performance mode (plugged in, disable sleep) 

# Run the benchmark harness
python benchmarks/enterprise/run_repro_bench.py \
  --dimensions 1024 4096 \
  --sizes 1000 10000 \
  --repeats 5 \
  --warmup 1 \
  --file-bytes 4096 \
  --outfile reports/benchmarks/hm_benchmark_report.json
```

Outputs:
- JSON report with environment, parameters, and P50/P95 values
- You can ingest this JSON into dashboards or generate Markdown summaries

## Interpreting Results
- Store throughput: expect significant improvement when using smaller payloads and looped “batches.” Values depend on Python overhead, backend, and CPU.
- doc_id recall: should be flat vs number of items (O(1) in N); scales with bytes returned.
- Search latency: grows with corpus size (O(n)) unless an ANN index is enabled.

## Why Numbers Vary
- Backend path: Python wrappers vs native core; HRR vs FFT route; exact-recall engines
- Build flags: `-O3 -march=native -ffast-math -funroll-loops -ftree-vectorize`
- Libraries: FFTW/Accelerate, BLAS; OS‐level vector libraries
- CPU power scaling and thermal throttling
- Cache warm-up and memory allocation behavior

## Reporting Ranges (Truth in Benchmarking)
- Always report ranges (P50/P95) rather than a single peak number
- Include environment (CPU, OS, Python), build flags, and corpus/dimension
- Example: “Store: 20k–140k ops/s on M2 (batch-like loops, M={1024,4096})”

## Next Steps (Optional)
- Add an ANN index for sub-linear semantic search
- Pin process to a performance core during measurement (platform-specific)
- Automate comparison runs (native-only vs Python API) with the same corpus

## GPU Acceleration (Apple Silicon)

For Mac Studio/MacBook with Apple Silicon, you can experiment with the experimental Metal path:

```bash
# Build experimental Metal extension (requires Xcode CLT)
python holographic-fs/native/holographic/build_metal.py

# Probe the Metal bridge from Python
python - << 'PY'
from holographicfs.backends.metal_backend import MetalHolographicBackend
m = MetalHolographicBackend()
print('available:', m.available())
print('sum:', m.vector_add_sum([1.0]*1024, [2.0]*1024))
PY
```

Notes:
- This path is a placeholder for compute throughput and isn’t wired into storage/search yet.
- Real holographic kernels (FFT via MPS, interference, similarity) should be added behind the same bridge.

## Caveats
- This harness exercises the Python-accessible API to keep results reproducible; there may be faster paths through direct native bindings. If you need the absolute peak, measure against native batch APIs under the same constraints and document the conditions.
