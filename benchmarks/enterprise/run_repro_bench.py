#!/usr/bin/env python3
"""Reproducible Holographic Memory Benchmarks (enterprise-grade).

Runs controlled benchmarks for:
- Store throughput (single vs looped batches)
- doc_id recall latency vs corpus size (O(1) claim check)
- Semantic search latency vs corpus size (expected O(n))

Outputs:
- JSON report with environment, parameters, and statistics
- Optional Markdown summary

Usage:
  python benchmarks/enterprise/run_repro_bench.py \
    --root ./.hm_bench --dimensions 1024 4096 \
    --sizes 1000 10000 --repeats 5 --warmup 1 \
    --outfile reports/benchmarks/hm_benchmark_report.json

Notes:
- This harness uses the Python-accessible HoloFS/Package API for consistency.
- For maximal throughput on native backends, ensure:
  * Native extensions are built with -O3/-march=native and vector libs (FFTW/Accelerate)
  * CPU power-scaling is minimized during the run (performance mode)
  * Caches are warmed (this harness includes warmup cycles)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
from math import floor
import string
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rand_bytes(n: int) -> bytes:
    return os.urandom(n)


def _rand_text(n: int) -> str:
    return "".join(random.choice(string.ascii_letters + string.digits + " ") for _ in range(n))


def _timer_ns(fn, repeat: int = 1) -> float:
    t0 = time.perf_counter_ns()
    for _ in range(max(1, int(repeat))):
        fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e9


def _env_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timestamp_ms": _now_ms(),
    }


@dataclass
class BenchParams:
    dimensions: List[int]
    sizes: List[int]
    repeats: int
    warmup: int
    file_bytes: int


@dataclass
class BenchResult:
    label: str
    dimension: int
    corpus_size: int
    metric: str
    values: List[float]
    p50: float
    p95: float


def bench_store_throughput(hm, N: int, file_bytes: int, repeats: int) -> List[float]:
    vals: List[float] = []
    payload = _rand_bytes(file_bytes)
    for _ in range(repeats):
        # store N items; compute ops/s
        dids: List[str] = []
        t0 = time.perf_counter_ns()
        for i in range(N):
            dids.append(hm.store(payload, filename=f"doc_{i}.bin"))
        dt = (time.perf_counter_ns() - t0) / 1e9
        vals.append(float(N) / dt if dt > 0 else float("nan"))
    return vals


def bench_recall_latency(hm, dids: List[str], repeats: int) -> List[float]:
    vals: List[float] = []
    for _ in range(repeats):
        did = random.choice(dids)
        t = _timer_ns(lambda: hm.retrieve(did))
        vals.append(t * 1000.0)  # ms
    return vals


def bench_search_latency(hm, dids: List[str], repeats: int) -> List[float]:
    vals: List[float] = []
    # Use a simple token that exists in filenames
    q = "doc_"
    for _ in range(repeats):
        t = _timer_ns(lambda: hm.search_semantic(q, top_k=5))
        vals.append(t * 1000.0)  # ms
    return vals


def _select_native(dim: int):
    """Try to load a native module with store_batch; return (name, ctor) or (None, None)."""
    mod_candidates = [
        ("holographic_native", "HolographicMemory"),
        ("holographic_wave_simd", "WaveMemory"),
    ]
    for mn, cls in mod_candidates:
        try:
            m = __import__(mn, fromlist=[cls])
            ctor = getattr(m, cls)
            return mn, ctor
        except Exception:  # noqa: BLE001
            continue
    return None, None


def bench_native_store_batch(dim: int, batch: int, repeats: int) -> Tuple[str, List[float]]:
    """Peak ops/s by calling native store_batch (bypasses Python loop overhead)."""
    name, ctor = _select_native(dim)
    if not ctor:
        return ("none", [])
    H = ctor(int(dim))
    vals: List[float] = []
    texts = ["x" * 64 for _ in range(batch)]
    # warmup once
    try:
        H.store_batch(texts, importance=1.0)
    except Exception:  # noqa: BLE001
        pass
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        H.store_batch(texts, importance=1.0)
        dt = (time.perf_counter_ns() - t0) / 1e9
        vals.append(float(batch) / dt if dt > 0 else float("nan"))
    return (name, vals)


def bench_gpu_batch_store(dim: int, batch: int, repeats: int) -> Tuple[str, List[float]]:
    """Benchmark GPU batch store performance via holographic_metal (Metal).

    Uses the new class-backed API if available; otherwise tries legacy bindings.
    Returns (name, values) where name is a backend tag.
    """
    try:
        # Prefer local build over site-packages
        import sys as _sys
        from pathlib import Path as _Path
        _root = _Path(__file__).resolve().parents[2]  # repo root
        _native = _root / "holographic-fs" / "native" / "holographic"
        if _native.exists():
            p = str(_native)
            if p not in _sys.path:
                _sys.path.insert(0, p)
            libp = str(_native / "lib.macosx-metal")
            if libp not in _sys.path:
                _sys.path.insert(0, libp)
        import holographic_metal as hm  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"GPU backend unavailable: {e}")
        return ("gpu_unavailable", [])

    # Prepare test data: batch of 64-float vectors
    batch_data = [[float(j) for j in range(64)] for _ in range(batch)]

    # Try class API first
    gpu_name = "gpu_metal"
    encode_fn = None
    try:
        if hasattr(hm, "MetalHolographicBackend"):
            be = hm.MetalHolographicBackend()  # type: ignore[attr-defined]
            if hasattr(be, "initialize") and not be.initialize():  # type: ignore[attr-defined]
                be = None
            if be is not None and hasattr(be, "batch_encode"):
                encode_fn = lambda: be.batch_encode(batch_data, int(dim))  # noqa: E731
    except Exception:
        encode_fn = None

    # Fallback: legacy module-level function
    if encode_fn is None and hasattr(hm, "gpu_batch_store"):
        encode_fn = lambda: hm.gpu_batch_store(batch_data, int(dim))  # type: ignore[attr-defined]  # noqa: E731
        gpu_name = "gpu_metal_legacy"

    if encode_fn is None:
        return ("gpu_unavailable", [])

    # Warmup
    try:
        _ = encode_fn()
    except Exception:  # noqa: BLE001
        return ("gpu_error", [])

    vals: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = encode_fn()
        dt = time.perf_counter() - t0
        if dt > 0:
            vals.append(float(batch) / dt)
    return (gpu_name, vals)


def bench_gpu_batch_store_fft(dim: int, batch: int, repeats: int) -> Tuple[str, List[float]]:
    """Benchmark GPU FFT-based batch store performance via holographic_gpu (Metal)."""
    try:
        # Prefer local build over site-packages
        import sys as _sys
        from pathlib import Path as _Path
        _root = _Path(__file__).resolve().parents[2]
        _native = _root / "holographic-fs" / "native" / "holographic"
        if _native.exists():
            p = str(_native)
            if p not in _sys.path:
                _sys.path.insert(0, p)
            libp = str(_native / "lib.macosx-metal")
            if libp not in _sys.path:
                _sys.path.insert(0, libp)
        import holographic_gpu as hg  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"GPU FFT backend unavailable: {e}")
        return ("gpu_unavailable", [])

    # Prepare test data: batch of 64-float vectors
    batch_data = [[float(j) for j in range(64)] for _ in range(batch)]

    gpu = hg.MetalHolographicBackend()  # type: ignore[attr-defined]
    if not gpu.available():
        return ("gpu_unavailable", [])
    vals: List[float] = []
    # Warmup
    try: _ = gpu.batch_encode_fft(batch_data, int(dim))
    except Exception: return ("gpu_fft_error", [])
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = gpu.batch_encode_fft(batch_data, int(dim))
        dt = time.perf_counter() - t0
        if dt > 0:
            vals.append(float(batch) / dt)
    return ("gpu_metal_fft", vals)


def run(params: BenchParams, out_json: Path) -> None:
    hm_pkg = None
    try:
        import holographic_memory as _hm_pkg  # type: ignore
        hm_pkg = _hm_pkg
    except Exception as e:  # noqa: BLE001
        # Allow GPU-only benchmarking when CPU package is unavailable
        print(f"warning: holographic_memory package not available, continuing with GPU-only: {e}")

    env = _env_info()
    results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for M in params.dimensions:
            hm = None
            if hm_pkg is not None:
                hm = hm_pkg.HolographicMemory(root=root)
                # Disable 3D exact-recall backend for capacity-independent measurements
                try:
                    hm.fs.mem.backend3d = None  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Warmup
                for _ in range(max(0, params.warmup)):
                    hm.store(_rand_bytes(params.file_bytes), filename="warm.bin")

            for N in params.sizes:
                # Store throughput (CPU) when available
                if hm is not None:
                    vals = bench_store_throughput(hm, N, params.file_bytes, params.repeats)
                    sv = sorted(vals)
                    mid = sv[len(sv)//2]
                    p95 = sv[min(len(sv)-1, floor(0.95*(len(sv)-1)))]
                    results.append(
                        asdict(
                            BenchResult(
                                label="store_throughput_ops_per_s",
                                dimension=M,
                                corpus_size=N,
                                metric="ops/s",
                                values=[round(v, 2) for v in vals],
                                p50=round(mid, 2),
                                p95=round(p95, 2),
                            )
                        )
                    )

                # Native peak throughput via store_batch (if available)
                n_name, n_vals = bench_native_store_batch(M, max(1000, N), params.repeats)
                if n_vals:
                    sv = sorted(n_vals)
                    mid = sv[len(sv)//2]
                    p95 = sv[min(len(sv)-1, floor(0.95*(len(sv)-1)))]
                    results.append(
                        asdict(
                            BenchResult(
                                label=f"native_store_batch_ops_per_s[{n_name}]",
                                dimension=M,
                                corpus_size=max(1000, N),
                                metric="ops/s",
                                values=[round(v, 2) for v in n_vals],
                                p50=round(mid, 2),
                                p95=round(p95, 2),
                            )
                        )
                    )

                # GPU batch store (Metal) throughput
                g_name, g_vals = bench_gpu_batch_store(M, max(1000, N), params.repeats)
                if g_vals:
                    sv = sorted(g_vals)
                    mid = sv[len(sv)//2]
                    p95 = sv[min(len(sv)-1, floor(0.95*(len(sv)-1)))]
                    results.append(
                        asdict(
                            BenchResult(
                                label=f"gpu_batch_store_ops_per_s[{g_name}]",
                                dimension=M,
                                corpus_size=max(1000, N),
                                metric="ops/s",
                                values=[round(v, 2) for v in g_vals],
                                p50=round(mid, 2),
                                p95=round(p95, 2),
                            )
                        )
                    )

                # GPU FFT batch store (Metal) throughput
                gf_name, gf_vals = bench_gpu_batch_store_fft(M, max(1000, N), params.repeats)
                if gf_vals:
                    sv = sorted(gf_vals)
                    mid = sv[len(sv)//2]
                    p95 = sv[min(len(sv)-1, floor(0.95*(len(sv)-1)))]
                    results.append(
                        asdict(
                            BenchResult(
                                label=f"gpu_batch_store_ops_per_s[{gf_name}]",
                                dimension=M,
                                corpus_size=max(1000, N),
                                metric="ops/s",
                                values=[round(v, 2) for v in gf_vals],
                                p50=round(mid, 2),
                                p95=round(p95, 2),
                            )
                        )
                    )

                # Doc_id recall latency (CPU) when available
                if hm is not None:
                    dids = [hm.store(_rand_bytes(params.file_bytes), filename=f"r_{i}.bin") for i in range(max(10, N // 10))]
                    vals = bench_recall_latency(hm, dids, params.repeats)
                    sv = sorted(vals)
                    mid = sv[len(sv)//2]
                    p95 = sv[min(len(sv)-1, floor(0.95*(len(sv)-1)))]
                    results.append(
                        asdict(
                            BenchResult(
                                label="doc_id_recall_latency_ms",
                                dimension=M,
                                corpus_size=len(dids),
                                metric="ms",
                                values=[round(v, 4) for v in vals],
                                p50=round(mid, 4),
                                p95=round(p95, 4),
                            )
                        )
                    )

                # Search latency vs N (expected O(n), CPU) when available
                if hm is not None:
                    vals = bench_search_latency(hm, dids, params.repeats)
                    sv = sorted(vals)
                    mid = sv[len(sv)//2]
                    p95 = sv[min(len(sv)-1, floor(0.95*(len(sv)-1)))]
                    results.append(
                        asdict(
                            BenchResult(
                                label="search_latency_ms",
                                dimension=M,
                                corpus_size=len(dids),
                                metric="ms",
                                values=[round(v, 4) for v in vals],
                                p50=round(mid, 4),
                                p95=round(p95, 4),
                            )
                        )
                    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"env": env, "params": asdict(params), "results": results}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ wrote benchmark report: {out_json}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible Holographic Memory Benchmarks")
    ap.add_argument("--root", default=".hm_bench", help="Working root (temp by default)")
    ap.add_argument("--dimensions", nargs="*", default=[1024], type=int)
    ap.add_argument("--sizes", nargs="*", default=[1000, 10000], type=int)
    ap.add_argument("--repeats", default=5, type=int)
    ap.add_argument("--warmup", default=1, type=int)
    ap.add_argument("--file-bytes", default=4096, type=int)
    ap.add_argument("--outfile", default="reports/benchmarks/hm_benchmark_report.json")
    args = ap.parse_args()

    params = BenchParams(
        dimensions=list(map(int, args.dimensions)),
        sizes=list(map(int, args.sizes)),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        file_bytes=int(args.file_bytes),
    )
    run(params, Path(args.outfile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
