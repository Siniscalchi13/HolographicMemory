#!/usr/bin/env python3
"""Wave ECC production stress test.

Exercises long-running encode/decode cycles with large logical sizes, error injection,
and optional concurrency. Streams chunks to avoid huge memory spikes.

Examples:
  PYTHONPATH=build_holo python scripts/stress_test_wave_ecc.py --sizes 100MB 500MB --duration 120 --workers 2

Outputs JSONL telemetry to logs/wave_ecc_stress.jsonl and prints periodic summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
    try:
        import holographic_gpu as hg  # type: ignore

        if hasattr(hg, "wave_ecc_encode") and hasattr(hg, "wave_ecc_decode"):
            return hg
    except Exception:
        pass
    # Fallback to memory helper
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


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def stream_chunks(total_bytes: int, chunk_size: int = 1 * 1024 * 1024) -> Iterable[bytes]:
    left = int(total_bytes)
    while left > 0:
        n = min(left, chunk_size)
        left -= n
        yield secrets.token_bytes(n)


def worker(idx: int, hg, sizes: List[int], redundancy: int, seed_base: int, stop_ts: float, out_path: Path, error_rate: float) -> None:
    # error_rate: fraction of bytes per chunk to flip (e.g., 1e-4)
    while time.time() < stop_ts:
        for size in sizes:
            processed = 0
            start = time.perf_counter()
            parity_total = 0
            ops = 0
            for payload in stream_chunks(size):
                t0 = time.perf_counter()
                parity = hg.wave_ecc_encode(payload, redundancy, seed_base)
                t1 = time.perf_counter()
                enc_ms = (t1 - t0) * 1000.0

                # error injection
                flips = max(1, int(len(payload) * error_rate))
                corrupted = bytearray(payload)
                for _ in range(flips):
                    pos = secrets.randbelow(len(corrupted))
                    corrupted[pos] ^= 0xFF

                t2 = time.perf_counter()
                corrected, _errors = hg.wave_ecc_decode(bytes(corrupted), parity, redundancy, seed_base)
                t3 = time.perf_counter()
                dec_ms = (t3 - t2) * 1000.0

                # Parity recheck ensures success/failure measurement
                ok = (hg.wave_ecc_encode(corrected, redundancy, seed_base) == parity)
                processed += len(payload)
                parity_total += len(parity or b"")
                ops += 1

                rec = {
                    "ts": time.time(),
                    "worker": idx,
                    "size_total": size,
                    "chunk_bytes": len(payload),
                    "redundancy": redundancy,
                    "encode_ms": enc_ms,
                    "decode_ms": dec_ms,
                    "parity_bytes": len(parity or b""),
                    "ok": bool(ok),
                }
                rec.update(_proc_sample())
                write_jsonl(out_path, rec)

                if time.time() >= stop_ts:
                    break
            dur = (time.perf_counter() - start)
            write_jsonl(out_path, {
                "ts": time.time(),
                "worker": idx,
                "summary": True,
                "size_total": size,
                "bytes_processed": processed,
                "ops": ops,
                "redundancy": redundancy,
                "throughput_mb_s": round((processed / (1024*1024)) / max(1e-6, dur), 3),
                "parity_overhead": round(parity_total / max(1, processed), 4) if processed else None,
                **_proc_sample(),
            })
            if time.time() >= stop_ts:
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", default=["100MB"], help="Logical sizes to simulate")
    ap.add_argument("--redundancy", type=int, default=3, help="Redundancy level (k)")
    ap.add_argument("--seed-base", type=int, default=42, help="Seed base")
    ap.add_argument("--workers", type=int, default=1, help="Concurrent worker threads")
    ap.add_argument("--duration", type=int, default=60, help="Total duration seconds (set 86400 for 24h)")
    ap.add_argument("--error-rate", type=float, default=1e-4, help="Fraction of bytes per chunk to flip")
    ap.add_argument("--out", default="logs/wave_ecc_stress.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    hg = _get_hg()
    sizes = [_human_to_bytes(s) for s in args.sizes]
    stop_ts = time.time() + max(1, int(args.duration))
    out_path = Path(args.out)

    print(f"Starting stress: sizes={args.sizes}, redundancy={args.redundancy}, workers={args.workers}, duration={args.duration}s")
    ths: List[threading.Thread] = []
    for i in range(max(1, int(args.workers))):
        t = threading.Thread(
            target=worker,
            kwargs={
                "idx": i,
                "hg": hg,
                "sizes": sizes,
                "redundancy": int(args.redundancy),
                "seed_base": int(args.seed_base),
                "stop_ts": stop_ts,
                "out_path": out_path,
                "error_rate": float(args.error_rate),
            },
            daemon=True,
        )
        t.start()
        ths.append(t)

    for t in ths:
        t.join()

    print(f"Stress test complete. Logs → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

