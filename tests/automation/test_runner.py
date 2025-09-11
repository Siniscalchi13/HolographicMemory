from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run(pytest_args: list[str] | None = None) -> int:
    args = ["pytest"]
    if pytest_args:
        args.extend(pytest_args)
    env = os.environ.copy()
    Path("tests/reports/html").mkdir(parents=True, exist_ok=True)
    Path("tests/reports/coverage").mkdir(parents=True, exist_ok=True)
    return subprocess.call(args, env=env)


if __name__ == "__main__":
    raise SystemExit(run())

