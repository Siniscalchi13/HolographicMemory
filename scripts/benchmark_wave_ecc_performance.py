#!/usr/bin/env python3
"""Wave ECC performance benchmarking.

Measures encode/decode throughput across sizes and redundancy levels on GPU (Metal).
Outputs JSON report under documentation/benchmarks/ and prints a concise summary.

Usage:
  PYTHONPATH=build_holo python scripts/benchmark_wave_ecc_performance.py \
    --sizes 1KB 10KB 100KB 1MB 10MB \
    --redundancy 1 3 5 7 \
    --iterations 3 \
    --output documentation/benchmarks/wave_ecc_performance_latest.json

Requires: holographic_gpu with Wave ECC (GPU-only), psutil (optional for RSS).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _human_to_bytes(s: str) -> int:
    t = s.strip().upper()
    if t.endswith("KB"):
        return int(float(t[:-2]) * 1024)
    if t.endswith("MB"):
        return int(float(t[:-2]) * 1024 * 1024)
    if t.endswith("GB"):
        return int(float(t[:-2]) * 1024 * 1024 * 1024)
    return int(t)


def _get_hg():
    # Prefer already-imported binding if present
    try:
        import holographic_gpu as hg  # type: ignore

        if hasattr(hg, "wave_ecc_encode") and hasattr(hg, "wave_ecc_decode"):
            return hg
    except Exception:
        pass
    # Fallback to memory helper which tries build_holo and dynamic loading
    try:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from services.holographic_memory.core.holographicfs.memory import (  # type: ignore
            _get_wave_binding,  # noqa: PLC2701
        )

        return _get_wave_binding()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Wave ECC binding not available. Build native GPU extension.") from exc


def _proc_sample() -> Dict[str, Any]:
    try:
        import psutil as ps  # type: ignore

        p = ps.Process()
        mi = p.memory_info()
        return {"rss_mb": round(mi.rss / (1024 * 1024), 2), "cpu_pct": p.cpu_percent(interval=0.0)}
    except Exception:
        return {"rss_mb": None, "cpu_pct": None}


def bench_case(hg, size: int, redundancy: int, seed_base: int, iterations: int = 3) -> Dict[str, Any]:
    import secrets

    encode_times: List[float] = []
    decode_times_clean: List[float] = []
    decode_times_corrupt: List[float] = []
    parity_sizes: List[int] = []

    for i in range(iterations):
        payload = secrets.token_bytes(size)
        t0 = time.perf_counter()
        parity = hg.wave_ecc_encode(payload, redundancy, seed_base)
        t1 = time.perf_counter()
        encode_times.append((t1 - t0) * 1000.0)
        parity_sizes.append(len(parity or b""))

        # Clean decode
        t2 = time.perf_counter()
        corrected, errors = hg.wave_ecc_decode(payload, parity, redundancy, seed_base)
        t3 = time.perf_counter()
        decode_times_clean.append((t3 - t2) * 1000.0)
        assert corrected == payload and int(errors) == 0

        # Light corruption (flip ~0.01% of bytes, at least 1)
        flips = max(1, size // 10000)
        corrupted = bytearray(payload)
        for _ in range(flips):
            pos = secrets.randbelow(size)
            corrupted[pos] ^= 0xFF
        t4 = time.perf_counter()
        corrected2, _errors2 = hg.wave_ecc_decode(bytes(corrupted), parity, redundancy, seed_base)
        t5 = time.perf_counter()
        decode_times_corrupt.append((t5 - t4) * 1000.0)
        # Parity recheck
        parity2 = hg.wave_ecc_encode(corrected2, redundancy, seed_base)
        assert bytes(parity2) == bytes(parity)

    avg = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {
        "size_bytes": int(size),
        "redundancy": int(redundancy),
        "seed_base": int(seed_base),
        "iterations": int(iterations),
        "encode_ms_avg": avg(encode_times),
        "decode_ms_avg_clean": avg(decode_times_clean),
        "decode_ms_avg_corrupt": avg(decode_times_corrupt),
        "parity_bytes_avg": avg(parity_sizes),
        "parity_overhead_avg": (avg(parity_sizes) / size) if size and avg(parity_sizes) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", default=["1KB", "10KB", "100KB", "1MB", "10MB"], help="Sizes to test")
    ap.add_argument("--redundancy", nargs="*", type=int, default=[1, 3, 5, 7], help="Redundancy levels")
    ap.add_argument("--iterations", type=int, default=3, help="Repeat count per case")
    ap.add_argument("--seed-base", type=int, default=42, help="Seed base")
    ap.add_argument("--output", default="documentation/benchmarks/wave_ecc_performance_latest.json", help="Output JSON path")
    args = ap.parse_args()

    hg = _get_hg()
    try:
        plats = getattr(hg, "available_platforms", lambda: [])()
    except Exception:
        plats = []
    gpu_ok = True if plats else False

    sizes = [_human_to_bytes(s) for s in args.sizes]
    results: List[Dict[str, Any]] = []
    started = time.time()
    proc0 = _proc_sample()

    for size in sizes:
        for r in args.redundancy:
            case = bench_case(hg, size=size, redundancy=r, seed_base=args.seed_base, iterations=args.iterations)
            case.update(_proc_sample())
            results.append(case)
            print(f"size={size:>10} bytes  R={r:>2}  enc_avg={case['encode_ms_avg']:.2f} ms  dec_clean={case['decode_ms_avg_clean']:.2f} ms  dec_corrupt={case['decode_ms_avg_corrupt']:.2f} ms  overhead={case['parity_overhead_avg']:.3f}")

    ended = time.time()
    out = {
        "started": started,
        "ended": ended,
        "duration_sec": round(ended - started, 3),
        "gpu_available": gpu_ok,
        "sizes": args.sizes,
        "redundancy_levels": args.redundancy,
        "iterations": int(args.iterations),
        "process_start": proc0,
        "process_end": _proc_sample(),
        "cases": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Also keep a timestamped report for history
    hist = out_path.parent / f"wave_ecc_performance_{int(started)}.json"
    hist.write_text(json.dumps(out), encoding="utf-8")

    print(f"\nSaved report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

