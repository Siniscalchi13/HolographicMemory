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
x = np.random.rand(8, 16).astype(np.float32)
enc = gpu.batch_encode_numpy(x, 16)
print('type(enc):', type(enc))
try:
    print('len(enc):', len(enc))
except Exception as e:
    print('no len:', e)
