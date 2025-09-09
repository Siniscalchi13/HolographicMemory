# HolographicMemory macOS App

This folder contains the native macOS app scaffold for running the HolographicMemory FastAPI server with Metal GPU acceleration (no Docker).

## What’s here

- `HolographicMemoryApp/` — XcodeGen project config, Swift sources, resources
- `HolographicMemoryApp/project.yml` — XcodeGen spec to generate the Xcode project
- `HolographicMemoryApp/Sources` — Swift AppKit app (menu bar + server process)
- `HolographicMemoryApp/Resources` — Python runner and where site-packages are embedded
- `scripts/macos/prepare_python_env.sh` — prepares an embedded Python.framework + site‑packages and builds the Metal GPU module

## Requirements

- macOS 13+
- Xcode 15+
- Homebrew: `brew install xcodegen cmake`

## One‑time setup

1) Prepare Python runtime and GPU module (Metal):

```bash
./scripts/macos/prepare_python_env.sh
```

This downloads Python 3.12 universal2, installs dependencies, and builds `holographic_gpu.cpython-*-darwin.so`. The runtime is placed under:

- `macos/HolographicMemoryApp/Runtime/Python.framework`
- `macos/HolographicMemoryApp/Runtime/Python/site-packages`

2) Generate the Xcode project:

```bash
cd macos/HolographicMemoryApp
xcodegen generate
```

3) Open and run:

```bash
open HolographicMemory.xcodeproj
# Press Run in Xcode (target: HolographicMemory, scheme auto‑generated)
```

On launch, the app starts a bundled Python FastAPI server and opens the browser to `http://localhost:8000`. The app places persistent data under `~/Library/Application Support/HolographicMemory`.

## App Store readiness

- Entitlements include client + server network, sandbox, and user‑selected read/write.
- Hardened runtime is enabled.
- For App Store submission, ensure code signing with a Developer ID / App Store profile.
- Consider embedding CPython via C‑API instead of spawning a process for stricter sandboxing; current approach uses an embedded Python binary inside `Python.framework` (acceptable for many apps but review latest App Review guidelines).

## Notes

- The Metal GPU backend is selected automatically by the `holographic_gpu` module built via CMake.
- The Python path is set to include `Resources/Python/site-packages`, `Resources/services`, and `Resources/holographic-fs` so your existing code runs unchanged.
- Optional packages like PyMuPDF are not strictly required; the API degrades gracefully without them.

## Troubleshooting

- If you see "Embedded Python not found", re‑run `./scripts/macos/prepare_python_env.sh`.
- If `holographic_gpu` fails to import, verify the module was built: check `macos/HolographicMemoryApp/Runtime/Python/site-packages/` for `holographic_gpu*.so`.
- To rebuild GPU module only:

```bash
# Inside the script, it uses CMake build dir at a temp path. Manually:
PYTHON=$(macos/HolographicMemoryApp/Runtime/Python.framework/Versions/3.12/bin/python3)
cmake -S holographic-fs/native/holographic -B /tmp/gpu \
  -DBUILD_METAL=ON -DBUILD_CUDA=OFF -DBUILD_ROCM=OFF \
  -Dpybind11_DIR=$($PYTHON -m pybind11 --cmake-dir)
cmake --build /tmp/gpu --config Release -j
cp /tmp/gpu/holographic_gpu*.so macos/HolographicMemoryApp/Runtime/Python/site-packages/
```

## Performance

- This runs natively on macOS with Metal Performance Shaders, matching the 451K ops/sec device‑side performance (subject to system state).
- The app launches the same FastAPI server you use in development, so behavior and endpoints remain unchanged.

