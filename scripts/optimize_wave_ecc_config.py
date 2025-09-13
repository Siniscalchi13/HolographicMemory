#!/usr/bin/env python3
"""Wave ECC configuration optimization.

Explores redundancy levels (k=1..10) across data types and error profiles.
Computes correction success vs storage overhead and generates recommendations.

Usage:
  PYTHONPATH=build_holo python scripts/optimize_wave_ecc_config.py \
    --types text binary compressed random --sizes 100KB 1MB \
    --redundancy-min 1 --redundancy-max 10 --trials 5 \
    --light 1e-4 --moderate 5e-4 --heavy 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
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


def synthesize(kind: str, size: int) -> bytes:
    import secrets

    if kind == "random":
        return secrets.token_bytes(size)
    if kind == "text":
        # Use README and build plan content to generate textual bytes
        parts: List[bytes] = []
        for p in [Path("README.md"), Path("HolographicMemory/HolographicMemory_Build_Plan.md")]:
            try:
                parts.append(p.read_bytes())
            except Exception:
                pass
        if not parts:
            parts = [b"HolographicMemory Wave ECC production text sample\n" * 1024]
        blob = b"\n\n".join(parts)
        # Repeat to desired size
        out = (blob * (size // max(1, len(blob)) + 1))[:size]
        return bytes(out)
    if kind == "compressed":
        # Compressed textual content to model near-entropy
        base = synthesize("text", min(size, 256 * 1024))
        comp = zlib.compress(base, level=9)
        return (comp * (size // max(1, len(comp)) + 1))[:size]
    if kind == "binary":
        # Mix of zeros and random blocks
        import secrets as _sec

        blk = b"\x00" * (size // 2) + _sec.token_bytes(size - size // 2)
        return blk
    raise ValueError(f"Unknown data type: {kind}")


def trial(hg, payload: bytes, k: int, r: int, error_rate: float) -> Tuple[bool, int, int]:
    import secrets

    parity = hg.wave_ecc_encode(payload, k, r)
    corrupted = bytearray(payload)
    flips = max(1, int(len(payload) * error_rate))
    for _ in range(flips):
        pos = secrets.randbelow(len(corrupted))
        corrupted[pos] ^= 0xFF
    corrected, _errs = hg.wave_ecc_decode(bytes(corrupted), parity, k, r)
    ok = (hg.wave_ecc_encode(corrected, k, r) == parity)
    return bool(ok), len(parity or b""), flips


def recommend(results: List[Dict[str, Any]], target_success: float = 0.99) -> Dict[str, Any]:
    # For each kind and error profile (light), pick minimal k achieving >= target_success
    recs: Dict[str, Any] = {}
    kinds = sorted(set(r["kind"] for r in results))
    profiles = sorted(set(r["profile"] for r in results))
    for kind in kinds:
        recs[kind] = {}
        for profile in profiles:
            subset = [r for r in results if r["kind"] == kind and r["profile"] == profile]
            # Group by k, average success
            byk: Dict[int, List[float]] = {}
            for r in subset:
                byk.setdefault(int(r["k"]), []).append(float(r["success_rate"]))
            best_k = None
            for k in sorted(byk):
                avg_succ = sum(byk[k]) / len(byk[k])
                if avg_succ >= target_success:
                    best_k = int(k)
                    break
            recs[kind][profile] = best_k
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=["text", "binary", "compressed", "random"], help="Data types")
    ap.add_argument("--sizes", nargs="*", default=["100KB"], help="Sizes to test per type")
    ap.add_argument("--redundancy-min", type=int, default=1)
    ap.add_argument("--redundancy-max", type=int, default=10)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--light", type=float, default=1e-4, help="Light error rate")
    ap.add_argument("--moderate", type=float, default=5e-4, help="Moderate error rate")
    ap.add_argument("--heavy", type=float, default=1e-3, help="Heavy error rate")
    ap.add_argument("--output", default="documentation/benchmarks/wave_ecc_config_recommendations.json")
    args = ap.parse_args()

    hg = _get_hg()
    sizes = [_human_to_bytes(s) for s in args.sizes]
    results: List[Dict[str, Any]] = []

    for kind in args.types:
        for size in sizes:
            payload = synthesize(kind, size)
            for k in range(int(args.redundancy_min), int(args.redundancy_max) + 1):
                for profile, rate in [("light", args.light), ("moderate", args.moderate), ("heavy", args.heavy)]:
                    succ = 0
                    parity_bytes = []
                    flips_total = 0
                    for _ in range(int(args.trials)):
                        ok, pbytes, flips = trial(hg, payload, k, int(args.seed_base), float(rate))
                        succ += 1 if ok else 0
                        parity_bytes.append(pbytes)
                        flips_total += flips
                    success_rate = succ / max(1, int(args.trials))
                    out = {
                        "kind": kind,
                        "size_bytes": int(size),
                        "k": int(k),
                        "profile": profile,
                        "error_rate": float(rate),
                        "trials": int(args.trials),
                        "success_rate": success_rate,
                        "parity_overhead": (sum(parity_bytes) / len(parity_bytes)) / size if parity_bytes else None,
                        "avg_flips": flips_total / max(1, int(args.trials)),
                    }
                    results.append(out)
                    print(f"{kind:10s} size={size:>8} k={k:>2} {profile:8s} success={success_rate:.3f} overhead={out['parity_overhead']:.3f}")

    recs = recommend(results)
    final = {
        "results": results,
        "recommendations": recs,
        "presets": {
            "balanced": {"redundancy": recs.get("text", {}).get("light") or 3},
            "high_reliability": {"redundancy": recs.get("random", {}).get("moderate") or 5},
            "low_overhead": {"redundancy": recs.get("compressed", {}).get("light") or 2},
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\nSaved recommendations → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

