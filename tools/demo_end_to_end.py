#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
from pathlib import Path

import requests


API = os.getenv("HOLO_API", "http://localhost:8000")
KEY = os.getenv("HOLO_API_KEY", "")


def post_file(fname: str, data: bytes, ctype: str = "text/plain") -> str:
    files = {"file": (fname, data, ctype)}
    r = requests.post(f"{API}/store", files=files, headers={"x-api-key": KEY})
    r.raise_for_status()
    doc_id = r.json()["doc_id"]
    print(f"Stored {fname} -> doc_id={doc_id}")
    return doc_id


def download(doc_id: str) -> bytes:
    r = requests.get(f"{API}/download/{doc_id}", headers={"x-api-key": KEY})
    r.raise_for_status()
    return r.content


def telemetry():
    r = requests.get(f"{API}/telemetry", headers={"x-api-key": KEY})
    r.raise_for_status()
    return r.json()


def main():
    # Store micro, microK8, and v4 files
    d1 = post_file("tiny.txt", b"tiny=true", "text/plain")
    d2 = post_file("small.txt", b"A" * 900, "text/plain")
    d3 = post_file("large.txt", (b"payload for v4\n" * 5000), "text/plain")

    # Download and verify sizes
    b1 = download(d1)
    b2 = download(d2)
    b3 = download(d3)
    print(f"Downloaded sizes: tiny={len(b1)} small={len(b2)} large={len(b3)}")

    # Show telemetry
    t = telemetry()
    print("Telemetry overall:", json.dumps(t.get("telemetry", {}).get("overall", {}), indent=2))
    print("Suggested D_k*:", json.dumps(t.get("suggested_dimensions", {}), indent=2))

    print("Visit dashboard:", f"{API}/dashboard")


if __name__ == "__main__":
    main()

