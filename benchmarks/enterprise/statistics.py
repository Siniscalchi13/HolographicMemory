from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class RunStats:
    iterations: int
    mean: float
    median: float
    stdev: float
    ci_low: float
    ci_high: float
    elapsed_s: float
    samples: List[float]


def _t_critical_975(df: int) -> float:
    """Approximate t critical value for 95% CI (two-tailed), df degrees of freedom.

    Uses SciPy if available, else a small lookup table + normal approx fallback.
    """
    try:  # Prefer precise value if SciPy is installed
        from scipy.stats import t as _t  # type: ignore

        return float(_t.ppf(0.975, df))
    except Exception:
        # Quick lookup for common dfs; beyond range use normal approx (z=1.96)
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
            25: 2.060,
            30: 2.042,
            40: 2.021,
            60: 2.000,
            120: 1.980,
        }
        if df in table:
            return table[df]
        if df < 120:
            # Linear interpolate between known points
            keys = sorted(table)
            for i in range(len(keys) - 1):
                if keys[i] < df < keys[i + 1]:
                    x0, y0 = keys[i], table[keys[i]]
                    x1, y1 = keys[i + 1], table[keys[i + 1]]
                    return y0 + (y1 - y0) * (df - x0) / (x1 - x0)
        return 1.96  # normal approx


class StatisticalValidation:
    """Run benchmarks repeatedly and compute 95% confidence intervals."""

    def run_benchmark_with_confidence(
        self,
        test_func: Callable[[], float],
        iterations: int = 100,
        warmup: int = 3,
    ) -> RunStats:
        if iterations <= 1:
            raise ValueError("iterations must be > 1 for CI computation")
        # Warmups (not recorded)
        for _ in range(max(0, warmup)):
            try:
                test_func()
            except Exception:
                # Warmup failures should not abort; continue to measurements
                pass

        samples: List[float] = []
        t0 = time.perf_counter()
        for _ in range(iterations):
            val = float(test_func())
            if math.isnan(val) or math.isinf(val):
                continue
            samples.append(val)
        t1 = time.perf_counter()

        if not samples:
            raise RuntimeError("Benchmark produced no valid samples")

        mean = statistics.fmean(samples)
        median = statistics.median(samples)
        stdev = statistics.pstdev(samples) if len(samples) <= 1 else statistics.stdev(samples)
        df = max(1, len(samples) - 1)
        tcrit = _t_critical_975(df)
        sem = stdev / math.sqrt(len(samples)) if len(samples) > 0 else float("inf")
        margin = tcrit * sem
        ci_low, ci_high = mean - margin, mean + margin
        return RunStats(
            iterations=len(samples),
            mean=mean,
            median=median,
            stdev=stdev,
            ci_low=ci_low,
            ci_high=ci_high,
            elapsed_s=(t1 - t0),
            samples=samples,
        )

    def validate_performance_claims(self, claims: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
        """Validate performance claims given observed CI bounds.

        claims map: name -> (observed_metric, required_threshold)
        Returns: name -> True/False whether claim is met
        """
        results: Dict[str, bool] = {}
        for name, (observed, required) in claims.items():
            results[name] = bool(observed >= required)
        return results

