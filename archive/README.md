# Archive Directory

This directory contains historical files, deprecated code, and temporary artifacts from the HolographicMemory project development.

## Directory Structure

### `test_files/`
Contains archived test files that were replaced by the main test suite:
- `test_rs_*.py` - Legacy Reed-Solomon ECC tests (replaced by Wave ECC)
- `test_wave_ecc_*.py` - Duplicate Wave ECC test variants (consolidated into `test_wave_ecc.py`)
- `test_rs_fix.cpp` - Legacy C++ RS implementation

### `old_docs/`
Historical documentation that no longer reflects the current architecture:
- Technical audit reports from before GPU-first implementation
- Outdated architecture documentation
- Legacy cleanup checklists

### `old_reports/`
Historical analysis and audit reports:
- Performance benchmarks from earlier phases
- Technical debt analysis reports
- Legacy system evaluation documents

### `old_scripts/`
Deprecated scripts and utilities:
- Legacy build scripts
- Old validation tools
- Historical demo applications

### Root Archive Files
- `codex_*.md` - Temporary files from Codex AI implementation sessions
- `codex_test_script.py` - Test script used during Phase 4 development

## Archive Policy

- **Purpose**: Preserve historical context and enable rollback if needed
- **Retention**: Files are kept for reference but should not be used for current development
- **Cleanup**: Review annually and remove files older than 2 years unless historically significant

## Last Updated
September 13, 2025 - Phase 4 completion cleanup
