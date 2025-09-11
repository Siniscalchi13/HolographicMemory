import sys, json
from pathlib import Path
root = Path('services/holographic-memory/core/native/holographic')
cands = []
if root.exists():
    for d in root.iterdir():
        if d.is_dir() and (d.name == 'build' or d.name.startswith('lib.')):
            cands.append(str(d.resolve()))
for p in cands:
    if p not in sys.path:
        sys.path.insert(0, p)
mods = {}
try:
    import holographic_gpu as hg
    mods['holographic_gpu'] = sorted([a for a in dir(hg) if not a.startswith('_')])
except Exception as e:
    mods['holographic_gpu_error'] = str(e)
try:
    import holographic_cpp as hc
    mods['holographic_cpp'] = sorted([a for a in dir(hc) if not a.startswith('_')])
except Exception as e:
    mods['holographic_cpp_error'] = str(e)
try:
    import holographic_cpp_3d as hc3
    mods['holographic_cpp_3d'] = sorted([a for a in dir(hc3) if not a.startswith('_')])
except Exception as e:
    mods['holographic_cpp_3d_error'] = str(e)
print(json.dumps(mods, indent=2))
