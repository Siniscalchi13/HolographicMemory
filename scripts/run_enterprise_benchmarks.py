#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest


def main() -> int:
    ap = argparse.ArgumentParser(description="Run enterprise benchmark suite")
    ap.add_argument("--suite", default="all", choices=["all", "holographic", "compression", "real", "competitors"],
                    help="Select subset of tests to run")
    ap.add_argument("--junit", type=str, default="", help="Optional JUnit XML output path")
    args, extra = ap.parse_known_args()

    test_paths = {
        "holographic": "benchmarks/enterprise/holographic_benchmarks.py::HolographicBenchmarkSuite",
        "compression": "benchmarks/enterprise/holographic_benchmarks.py::CompressionBenchmarks",
        "real": "benchmarks/enterprise/holographic_benchmarks.py::RealWorldBenchmarks",
        "competitors": "benchmarks/enterprise/holographic_benchmarks.py::CompetitorBenchmarkSuite",
    }
    if args.suite == "all":
        path = "benchmarks/enterprise/holographic_benchmarks.py"
    else:
        path = test_paths[args.suite]

    pytest_args = [path, "-q"] + extra
    if args.junit:
        pytest_args += ["--junitxml", args.junit]

    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())

