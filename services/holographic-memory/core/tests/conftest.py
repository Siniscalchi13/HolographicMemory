import os
import sys
from pathlib import Path

# Ensure build_holo is first in path for Wave ECC binding
here = Path(__file__).resolve()
repo_root = here.parents[4]
# Prefer build_holo for the GPU binding
build_holo = str(repo_root / 'build_holo')
if build_holo not in sys.path:
    sys.path.insert(0, build_holo)
# Also ensure the core package path is importable for `holographicfs`
core_path = str(repo_root / 'services' / 'holographic-memory' / 'core')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

# Import Wave ECC binding early to ensure it's available
import holographic_gpu as hg  # noqa: F401
