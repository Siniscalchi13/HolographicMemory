import sys, json
from pathlib import Path
root = Path('services/holographic-memory/core/native/holographic')
for d in root.iterdir():
    if d.is_dir() and (d.name == 'build' or d.name.startswith('lib.')):
        p=str(d.resolve())
        if p not in sys.path:
            sys.path.insert(0,p)
import holographic_gpu as hg
obj = hg.HolographicGPU()
if hasattr(obj,'initialize'):
    try:
        obj.initialize('metal')
    except Exception:
        try:
            obj.initialize()
        except Exception:
            pass
attrs = {}
for a in dir(obj):
    if a.startswith('_'): continue
    try:
        v = getattr(obj,a)
        if callable(v):
            attrs[a] = 'callable'
        else:
            attrs[a] = type(v).__name__
    except Exception as e:
        attrs[a] = f'error:{e}'
print(json.dumps(attrs, indent=2))
