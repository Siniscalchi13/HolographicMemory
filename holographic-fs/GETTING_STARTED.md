Getting Started in 5 Minutes

1) Install (editable from this repo)

- cd holographic-fs
- python -m pip install -e .

2) Initialize a store

- holo init ~/Documents --grid-size 64

3) Store and search

- holo store ~/Documents/example.pdf
- holo search "example"

4) Recall

- holo recall --doc <DOC_ID> --out ./example.pdf
- or: holo recall "example" --original

5) Stats

- holo stats --root ~/Documents

Benchmarks

- python holographic-fs/benchmarks/compare_gzip.py

Notes

- The preview uses a Python backend and a fixed-size wave field. Choose a grid size appropriate for demo content, not full disks.
- Compression reported reflects on-disk wave and index size versus total original size tracked in the index.
