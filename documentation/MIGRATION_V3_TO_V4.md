# Migration Guide: v3 JSON .hwp → v4 Binary (Sparse/Micro)

Version: 1.0

## Overview
This guide explains how to convert existing v3 JSON `.hwp` files (with base64 payloads) to compact v4 binary artifacts using the migration tool.

## Tool
- Script: `scripts/migrate_hwp_v3_to_v4.py`
- Prerequisites: API/HoloFS environment available; the tool calls into HoloFS to recompute wave vectors.

## Usage
```
python3 scripts/migrate_hwp_v3_to_v4.py
```
The script scans `${HLOG_DATA_DIR}/patterns/*.hwp`. If a file is JSON v3, it recomputes the adaptive wave vector and writes a v4 binary replacement (single "knowledge" layer by default).

## Notes
- Original bytes remain reconstructable from the 3D backend.
- For Vault items, ensure they are not converted to v4 sketches; Vault policy mandates micro only with random IDs.

