#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.api.hwp_v4 import build_sparse_layer, write_hwp_v4  # type: ignore
from holographicfs.memory import mount  # type: ignore


def main() -> int:
    root = Path(os.getenv("HOLO_ROOT", "./data")).expanduser().resolve()
    patterns = Path(os.getenv("HLOG_DATA_DIR", str(root / "holographic_memory"))) / "patterns"
    if not patterns.exists():
        print(f"No patterns directory: {patterns}")
        return 1
    fs = mount(root)
    converted = 0
    skipped = 0
    for p in patterns.glob("*.hwp"):
        try:
            txt = p.read_text(encoding="utf-8")
            j = json.loads(txt)
        except Exception:
            # likely already v4 binary
            skipped += 1
            continue
        doc_id = j.get("doc_id")
        orig = j.get("original", {})
        fname = orig.get("filename", p.stem)
        ctype = orig.get("content_type", "")
        size = int(orig.get("size", 0))
        if not doc_id:
            print(f"Skipping {p.name}: no doc_id")
            skipped += 1
            continue
        # Recompute wave vector for sparse v4
        try:
            wave = fs.mem.get_real_wave_data(doc_id)
            amps = list(map(float, wave.get("amplitudes", []) or []))
            phs = list(map(float, wave.get("phases", []) or []))
            dim = int(wave.get("dimension", 0) or len(amps))
            layer = build_sparse_layer("knowledge", amps, phs, top_k=int(os.getenv("HOLO_TOPK", "32") or 32))
            # Write to temp, then replace
            tmp = p.with_suffix(".hwp.tmp")
            write_hwp_v4(tmp, doc_id=doc_id, filename=fname, original_size=size, content_type=ctype, dimension=dim, layers=[layer])
            p.unlink()
            tmp.rename(p)
            converted += 1
            print(f"Converted {p.name} -> v4 binary")
        except Exception as e:
            print(f"Failed to convert {p.name}: {e}")
            skipped += 1
    print(f"Done. converted={converted} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

