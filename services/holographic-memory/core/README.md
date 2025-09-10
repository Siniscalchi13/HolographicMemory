HolographicFS

Core idea: a local, privacy-first holographic memory store with a simple CLI.

Quickstart

- Create a new store: `holo init ~/Documents --grid-size 64`
- Store a file: `holo store ~/Documents/example.pdf`
- Search by name: `holo search "example"`
- Recall by id: `holo recall --doc <DOC_ID> --out ./example.recalled.pdf`
- See stats: `holo stats`

Notes

- Engine: C++ holographic memory only (no Python implementation).
- Build C++ locally: `make cpp` from repo root (requires pybind11 and FFTW/Accelerate).
- Byte-level file recall is not exposed by the current C++ engine; CLI `recall` will raise until implemented.
- Capacity scales with `grid_size^3`. Large directories require larger grids; start small for demos.
- Compression metrics shown are based on actual on-disk state size vs total original size in the index.
