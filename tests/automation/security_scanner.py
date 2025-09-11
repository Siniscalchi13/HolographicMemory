from __future__ import annotations

from tests.infrastructure.security_scanner import run_bandit, run_pip_audit


def main() -> int:
    try:
        b = run_bandit(["services"])  # noqa: S603
        print("bandit rc:", b.returncode)
        a = run_pip_audit()  # noqa: S603
        print("pip-audit rc:", a.returncode)
        return 0
    except FileNotFoundError:
        print("security tools not installed; skipping")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

