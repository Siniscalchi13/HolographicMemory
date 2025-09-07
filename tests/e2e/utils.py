from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import requests


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upload_file(base: str, path: Path) -> Tuple[str, str]:
    files = {"file": (path.name, path.read_bytes())}
    r = requests.post(f"{base}/store", files=files, timeout=30)
    r.raise_for_status()
    doc_id = r.json()["doc_id"]
    return doc_id, path.name


def list_files(base: str):
    r = requests.get(f"{base}/list", timeout=10)
    r.raise_for_status()
    return r.json().get("results", [])


def download_doc(base: str, doc_id: str) -> bytes:
    r = requests.get(f"{base}/download/{doc_id}", timeout=30)
    r.raise_for_status()
    return r.content

