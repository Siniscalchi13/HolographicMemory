from __future__ import annotations

import re
import requests


def test_root_serves_web_ui(base_url: str):
    r = requests.get(f"{base_url}/", timeout=10)
    assert r.ok
    assert "<title>Holographic Memory</title>" in r.text


def test_stats_and_list_api_present(base_url: str):
    s = requests.get(f"{base_url}/stats").json()
    assert "files_indexed" in s
    lst = requests.get(f"{base_url}/list").json()
    assert isinstance(lst.get("results", []), list)

