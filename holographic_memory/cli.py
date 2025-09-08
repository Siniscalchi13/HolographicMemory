from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__
from .memory import HolographicMemory


def main() -> int:
    parser = argparse.ArgumentParser(prog="holo", description="HolographicMemory CLI")
    parser.add_argument("command", choices=["version", "store", "retrieve"], help="Command to run")
    parser.add_argument("path", nargs="?", help="Path to file (store) or doc_id (retrieve)")
    parser.add_argument("--root", dest="root", default=None, help="Data root directory")
    parser.add_argument("--out", dest="out", default=None, help="Output path for retrieve")
    args = parser.parse_args()

    if args.command == "version":
        print(__version__)
        return 0

    hm = HolographicMemory(root=args.root)
    if args.command == "store":
        if not args.path:
            print("error: path required", file=sys.stderr)
            return 2
        p = Path(args.path)
        if not p.exists() or not p.is_file():
            print(f"error: no such file: {p}", file=sys.stderr)
            return 2
        did = hm.store(p.read_bytes(), filename=p.name)
        print(did)
        return 0
    if args.command == "retrieve":
        if not args.path:
            print("error: doc_id required", file=sys.stderr)
            return 2
        data = hm.retrieve(args.path)
        out = Path(args.out) if args.out else (Path.cwd() / f"{args.path}.bin")
        out.write_bytes(data)
        print(str(out))
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

