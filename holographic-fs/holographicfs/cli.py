from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .memory import mount


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser()
    fs = mount(root, grid_size=args.grid_size, state_dir=args.state_dir)
    # Create state dir and write a manifest
    fs.state_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "root": str(fs.root),
        "grid_size": fs.grid_size,
        "state_dir": str(fs.state_dir),
        "created_by": "HolographicFS",
    }
    (fs.state_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Initialized HoloFS at {fs.state_dir}")
    return 0


def cmd_store(args: argparse.Namespace) -> int:
    root = Path(args.root or Path.cwd())
    fs = mount(root, grid_size=args.grid_size, state_dir=args.state_dir)
    path = Path(args.path).expanduser()
    doc_id = fs.store(path, force=args.force)
    print(doc_id)
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    root = Path(args.root or Path.cwd())
    fs = mount(root, grid_size=args.grid_size, state_dir=args.state_dir)
    out = Path(args.out).expanduser() if args.out else None
    try:
        p = fs.recall(args.query_or_doc, out=out, original=args.original)
        print(str(p))
        return 0
    except Exception as e:
        print(f"error: {e}")
        return 2


def cmd_search(args: argparse.Namespace) -> int:
    root = Path(args.root or Path.cwd())
    fs = mount(root, grid_size=args.grid_size, state_dir=args.state_dir)
    # Prefer index-based search by filename/path; fallback to content search
    results = fs.search_index(args.query)
    if results:
        for row in results[: args.k]:
            # Support tuples with or without mtime
            doc_id, path, size = row[0], row[1], row[2]
            print(f"{doc_id}\t{size}\t{path}")
        return 0
    # Fallback to holographic content search (if available)
    try:
        hits = fs.search(args.query, k=args.k)
        for doc_id, score, text in hits:
            print(f"{doc_id}\t{score:.3f}\t{text}")
        return 0
    except Exception:
        return 0


def cmd_stats(args: argparse.Namespace) -> int:
    root = Path(args.root or Path.cwd())
    fs = mount(root, grid_size=args.grid_size, state_dir=args.state_dir)
    s = fs.stats()
    print(json.dumps(s, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="holo", description="HolographicFS CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=str, default=str(Path.cwd()), help="Root directory for this store (default: cwd)")
    common.add_argument("--state-dir", type=str, default=None, help="Override state directory (default: <root>/.holofs)")
    common.add_argument("--grid-size", type=int, default=32, help="Holographic grid size (default: 32)")

    # init
    p_init = sub.add_parser("init", parents=[common], help="Initialize a HoloFS store")
    p_init.add_argument("root", type=str, help="Directory to manage with HoloFS")
    p_init.set_defaults(func=cmd_init)

    # store
    p_store = sub.add_parser("store", parents=[common], help="Store a file in holographic memory")
    p_store.add_argument("path", type=str, help="Path to file")
    p_store.add_argument("--force", action="store_true", help="Force re-store even if unchanged")
    p_store.set_defaults(func=cmd_store)

    # recall
    p_recall = sub.add_parser("recall", parents=[common], help="Recall a file by doc id or name")
    p_recall.add_argument("query_or_doc", type=str, help="Doc id (sha256) or filename substring")
    p_recall.add_argument("--out", type=str, help="Output path (default: <doc>.bin or original with --original)")
    p_recall.add_argument("--original", action="store_true", help="Write to original indexed path if found")
    p_recall.set_defaults(func=cmd_recall)

    # search
    p_search = sub.add_parser("search", parents=[common], help="Search by filename substring")
    p_search.add_argument("query", type=str, help="Query string")
    p_search.add_argument("-k", type=int, default=5, help="Top-K")
    p_search.set_defaults(func=cmd_search)

    # stats
    p_stats = sub.add_parser("stats", parents=[common], help="Show store stats and compression estimate")
    p_stats.set_defaults(func=cmd_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
