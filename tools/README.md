# HolographicMemory Tools

This directory contains development tools, scripts, and utilities for the HolographicMemory system.

## Tool Categories

### Development Tools
- **`demo_end_to_end.py`** - End-to-end demonstration script
- **`migrate_hwp_v3_to_v4.py`** - Migration utility for HWP format
- **`migrate_hwp.py`** - General HWP migration tools
- **`update_version.py`** - Version management utility

### Validation & Testing
- **`validate_accuracy.py`** - Accuracy validation tests
- **`validate_build_matrix.sh`** - Build matrix validation
- **`test_makefile.sh`** - Makefile testing utility

### Setup & Configuration
- **`setup_auth.py`** - Authentication setup
- **`setup_consumer_project.py`** - Consumer project setup
- **`run_enterprise_benchmarks.py`** - Enterprise benchmark runner

### macOS Specific
- **`macos/dev_server.sh`** - macOS development server
- **`macos/prepare_python_env.sh`** - Python environment preparation

## Usage

### Running Tools
```bash
# Run end-to-end demo
python tools/demo_end_to_end.py

# Validate accuracy
python tools/validate_accuracy.py

# Run enterprise benchmarks
python tools/run_enterprise_benchmarks.py
```

### Development Setup
```bash
# Prepare macOS environment
./tools/macos/prepare_python_env.sh

# Start development server
./tools/macos/dev_server.sh
```

## Tool Development

When adding new tools:
1. Follow the existing naming conventions
2. Add proper error handling and logging
3. Include usage documentation in the tool
4. Update this README with tool descriptions
