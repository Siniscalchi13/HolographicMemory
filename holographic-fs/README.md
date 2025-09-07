HolographicFS (Preview)

Core idea: a local, privacy-first holographic memory store with a simple CLI.

Quickstart

- Create a new store: `holo init ~/Documents --grid-size 64`
- Store a file: `holo store ~/Documents/example.pdf`
- Search by name: `holo search "example"`
- Recall by id: `holo recall --doc <DOC_ID> --out ./example.recalled.pdf`
- See stats: `holo stats`

Notes

- This preview uses the Python holographic backend and persists state under `.holofs/` in the selected directory by default.
- Capacity scales with `grid_size^3`. Large directories require larger grids; start small for demos.
- Compression metrics shown are based on actual on-disk state size vs total original size in the index.

