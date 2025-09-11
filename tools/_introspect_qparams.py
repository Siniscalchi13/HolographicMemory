import sys
from pathlib import Path
root = Path('services/holographic-memory/core/native/holographic')
for d in root.iterdir():
    if d.is_dir() and (d.name == 'build' or d.name.startswith('lib.')):
        p=str(d.resolve())
        if p not in sys.path:
            sys.path.insert(0,p)
import holographic_gpu as hg
qp = hg.QuantizationParams()
attrs = [a for a in dir(qp) if not a.startswith('_')]
vals = {}
for a in attrs:
    try:
        vals[a]=getattr(qp,a)
    except Exception as e:
        vals[a]=f'error:{e}'
print('attrs:',attrs)
print('vals:',vals)
