#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import base64
from pathlib import Path
from typing import Optional


def _mount_fs(root: Path, grid: int = 64):
    # Add package path for local imports
    pkg = Path(__file__).resolve().parents[1] / "holographic-fs"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from holographicfs.memory import mount  # type: ignore
    return mount(root, grid_size=grid)


def quantize_and_pack(amps: list[float], phs: list[float], dtype: str = "f16") -> dict:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy required for quantization") from e
    dt = np.float16 if dtype.lower() in ("f16", "float16", "half") else np.float32
    a = np.asarray(amps, dtype=dt) if amps else np.asarray([], dtype=dt)
    p = np.asarray(phs, dtype=dt) if phs else np.asarray([], dtype=dt)
    return {
        "dtype": "float16" if dt is np.float16 else "float32",
        "endian": "little",
        "count": int(a.size),
        "amplitudes_b64": base64.b64encode(a.tobytes()).decode("ascii") if a.size else "",
        "phases_b64": base64.b64encode(p.tobytes()).decode("ascii") if p.size else "",
        "source": "holographic_engine",
    }


def ensure_engine_mapping(fs, doc_id: str, filename: str, size: int) -> None:
    try:
        meta = f"filename:{Path(filename).name}\nsize:{int(size)}\nsha256:{doc_id}\n"
        eng_id = fs.mem.backend.store(meta)  # type: ignore[attr-defined]
        mapping = {"doc_id": doc_id, "engine_id": eng_id, "filename": filename, "size": int(size)}
        # Volatile map
        fs.mem.backend.store_response_hrr(f"{doc_id}#engine_mapping", json.dumps(mapping))  # type: ignore[attr-defined]
        # Persistent map
        mpath = Path(fs.state_dir) / "engine_map.json"  # type: ignore[attr-defined]
        db = {}
        if mpath.exists():
            try:
                db = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                db = {}
        db[str(doc_id)] = mapping
        mpath.write_text(json.dumps(db, indent=2), encoding="utf-8")
    except Exception:
        pass


def migrate_file(fs, hwp_path: Path, dtype: str = "f16", dry_run: bool = False, verify: bool = False) -> dict:
    info = {"path": str(hwp_path), "before": hwp_path.stat().st_size, "after": None, "changed": False}
    # Skip binary .hwp (native engine pattern/current snapshots)
    try:
        with hwp_path.open("rb") as f:
            magic = f.read(8)
            if magic == b"WAVEV001":
                info["after"] = info["before"]
                info["changed"] = False
                return info
    except Exception:
        pass
    try:
        j = json.loads(hwp_path.read_text(encoding="utf-8"))
    except Exception as e:
        info["error"] = f"json parse failed: {e}"
        return info

    original = j.get("original", {})
    filename = original.get("filename", hwp_path.stem)
    size = int(original.get("size", 0))
    doc_id = j.get("doc_id") or original.get("sha256")

    # 1) If base64 payload exists, move it into engine recall and strip from JSON
    if isinstance(j.get("data"), str) and j.get("data"):
        try:
            raw = base64.b64decode(j["data"].encode("ascii"), validate=False)
        except Exception as e:
            info["error"] = f"base64 decode failed: {e}"
            return info
        # Resolve doc_id
        import hashlib
        if not doc_id:
            doc_id = hashlib.sha256(raw).hexdigest()
        # Store into 3D engine if available, else fallback to HRR/disk
        try:
            if getattr(fs.mem, "backend3d", None) is not None:
                try:
                    fs.mem.backend3d.store_bytes(raw, doc_id)  # type: ignore[attr-defined]
                except Exception:
                    # Force legacy persistence: HRR base64 + disk-backed file
                    try:
                        b64 = base64.b64encode(raw).decode("ascii")
                        if hasattr(fs.mem.backend, "store_response_hrr"):
                            fs.mem.backend.store_response_hrr(f"{doc_id}#data", b64)  # type: ignore[attr-defined]
                            fs.mem.backend.store_response_hrr(
                                f"{doc_id}#manifest",
                                json.dumps({"doc_id": doc_id, "size": len(raw), "filename": Path(filename).name, "type": "base64_direct"}),
                            )  # type: ignore[attr-defined]
                        resp_dir = Path(fs.state_dir) / "responses" / doc_id  # type: ignore[attr-defined]
                        resp_dir.mkdir(parents=True, exist_ok=True)
                        (resp_dir / "data.bin").write_bytes(raw)
                        (resp_dir / "meta.json").write_text(json.dumps({"filename": Path(filename).name, "size": len(raw)}), encoding="utf-8")
                    except Exception:
                        raise
            else:
                # Legacy persistence when 3D engine is absent
                b64 = base64.b64encode(raw).decode("ascii")
                if hasattr(fs.mem.backend, "store_response_hrr"):
                    fs.mem.backend.store_response_hrr(f"{doc_id}#data", b64)  # type: ignore[attr-defined]
                    fs.mem.backend.store_response_hrr(
                        f"{doc_id}#manifest",
                        json.dumps({"doc_id": doc_id, "size": len(raw), "filename": Path(filename).name, "type": "base64_direct"}),
                    )  # type: ignore[attr-defined]
                resp_dir = Path(fs.state_dir) / "responses" / doc_id  # type: ignore[attr-defined]
                resp_dir.mkdir(parents=True, exist_ok=True)
                (resp_dir / "data.bin").write_bytes(raw)
                (resp_dir / "meta.json").write_text(json.dumps({"filename": Path(filename).name, "size": len(raw)}), encoding="utf-8")
        except Exception as e:
            info["error"] = f"engine store failed: {e}"
            return info
        # Optional verify
        if verify:
            try:
                rb = fs.mem.retrieve_bytes(doc_id)
                if rb != raw:
                    info["error"] = "verify mismatch after engine store"
                    return info
            except Exception as e:
                info["error"] = f"verify failed: {e}"
                return info
        # Ensure mapping for wave access
        ensure_engine_mapping(fs, doc_id, filename, size if size else len(raw))
        # Strip base64 from JSON
        if not dry_run:
            j.pop("data", None)
            j.pop("encoding", None)
            j["doc_id"] = doc_id
            info["changed"] = True

    # 2) Quantize wave arrays in JSON if present as lists
    hw = j.get("holographic_wave", {}) or {}
    # If already packed (amplitudes_b64 present), possibly re-encode dtype
    amps_b64 = hw.get("amplitudes_b64")
    phs_b64 = hw.get("phases_b64")
    amps_list = hw.get("amplitudes")
    phs_list = hw.get("phases")
    if amps_list is not None or phs_list is not None:
        packed = quantize_and_pack(amps_list or [], phs_list or [], dtype=dtype)
        if not dry_run:
            j["holographic_wave"] = packed
            info["changed"] = True
    elif amps_b64 is not None and phs_b64 is not None:
        # Re-encode if dtype change requested
        if isinstance(amps_list, list) or isinstance(phs_list, list):
            # fallthrough handled above
            pass
        else:
            # Keep as is unless dtype change requested and lists are available (we cannot change without source floats)
            pass
    else:
        # Try to fetch wave from engine if we have doc_id
        if doc_id:
            try:
                wd = fs.mem.get_real_wave_data(doc_id)
                packed = quantize_and_pack(wd.get("amplitudes", []), wd.get("phases", []), dtype=dtype)
                if not dry_run:
                    j["holographic_wave"] = packed
                    info["changed"] = True
            except Exception:
                pass

    # 3) Update version
    if not dry_run and info["changed"]:
        j["version"] = 3
        hwp_path.write_text(json.dumps(j, indent=2), encoding="utf-8")
        info["after"] = hwp_path.stat().st_size
    else:
        info["after"] = info["before"]
    return info


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Migrate .hwp files to wave-only quantized format and remove base64 data")
    ap.add_argument("--root", default=os.getenv("HOLO_ROOT", "./data"), help="FS root where .holofs lives")
    ap.add_argument("--hlog", default=os.getenv("HLOG_DATA_DIR", None), help="Holographic data dir (defaults under root)")
    ap.add_argument("--dtype", default=os.getenv("HOLO_WAVE_DTYPE", "f16"), choices=["f16", "f32", "float16", "float32"], help="Quantization dtype")
    ap.add_argument("--verify", action="store_true", help="Verify recall equals original for files with base64 payload")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    fs = _mount_fs(root)
    # Determine holographic storage dir
    if args.hlog:
        hlog = Path(args.hlog).expanduser().resolve()
    else:
        # mirror native default
        hlog = Path(os.getenv("HLOG_DATA_DIR", str(root / "holographic_memory")))
    pat = hlog / "patterns"
    pat.mkdir(parents=True, exist_ok=True)

    files = sorted(pat.glob("*.hwp"))
    if not files:
        print(f"No .hwp files under {pat}")
        return 0
    total_before = 0
    total_after = 0
    changed = 0
    for f in files:
        info = migrate_file(fs, f, dtype=args.dtype, dry_run=args.dry_run, verify=args.verify)
        total_before += int(info.get("before", 0) or 0)
        total_after += int(info.get("after", info.get("before", 0)) or 0)
        if info.get("changed"):
            changed += 1
        line = f"{f.name}: before={info.get('before')} after={info.get('after')} changed={info.get('changed', False)}"
        if info.get("error"):
            line += f" ERROR={info['error']}"
        print(line)
    print(f"Summary: files={len(files)} changed={changed} total_before={total_before} total_after={total_after} delta={total_after-total_before}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
