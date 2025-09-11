import sys
from pathlib import Path
root = Path('services/holographic-memory/core/native/holographic')
for d in root.iterdir():
    if d.is_dir() and (d.name == 'build' or d.name.startswith('lib.')):
        p=str(d.resolve())
        if p not in sys.path:
            sys.path.insert(0,p)
import holographic_gpu as hg
m = hg.PerformanceMetrics()
print([a for a in dir(m) if not a.startswith('_')])
