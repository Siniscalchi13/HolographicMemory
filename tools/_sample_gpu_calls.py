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
print('layers_initialized:', getattr(gpu,'layers_initialized', None))
print('type(get_layer_stats):', type(gpu.get_layer_stats()))
print('type(optimize_layer_dimensions):', type(gpu.optimize_layer_dimensions()))
print('type(metrics):', type(gpu.metrics()))
# Try a small batch_encode_numpy
x = np.random.rand(8, 16).astype(np.float32)
enc = gpu.batch_encode_numpy(x, 16)
print('enc shape:', getattr(enc, 'shape', None))
# Try quantization params if present
if hasattr(hg, 'QuantizationParams'):
    params = hg.QuantizationParams(phase_bits=8, amplitude_bits=8)
    q = gpu.gpu_holographic_quantize_with_validation(x, x, 0, params)
    print('quantized ok:', q is not None)
