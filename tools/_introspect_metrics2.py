import sys
from pathlib import Path
root = Path('services/holographic-memory/core/native/holographic')
for d in root.iterdir():
    if d.is_dir() and (d.name == 'build' or d.name.startswith('lib.')):
        p=str(d.resolve())
        if p not in sys.path:
            sys.path.insert(0,p)
import holographic_gpu as hg
import numpy as np

gpu = hg.HolographicGPU()
try:
    gpu.initialize('metal')
except Exception:
    try:
        gpu.initialize()
    except Exception:
        pass
x = np.random.rand(4, 8).astype(np.float32)
_ = gpu.batch_encode_numpy(x, 8)
pm = gpu.metrics()
attrs = [a for a in dir(pm) if not a.startswith('_')]
print('attrs:', attrs)
vals = {}
for a in attrs:
    try:
        vals[a] = getattr(pm,a)
    except Exception as e:
        vals[a] = f'error:{e}'
print('values:', vals)
