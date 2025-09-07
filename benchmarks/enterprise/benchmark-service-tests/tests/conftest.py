import sys
from pathlib import Path

# Ensure project root is on sys.path so `import services...` works in tests
tests_dir = Path(__file__).resolve().parent
services_dir = tests_dir.parents[2]  # .../services
repo_root = tests_dir.parents[3]     # .../TAI
for p in (str(services_dir), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)
