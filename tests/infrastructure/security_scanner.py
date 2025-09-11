from __future__ import annotations

import subprocess
from typing import List


def run_bandit(paths: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bandit", "-q", "-r", *paths], capture_output=True, text=True)


def run_pip_audit() -> subprocess.CompletedProcess[str]:
    return subprocess.run(["pip-audit", "-q"], capture_output=True, text=True)

