#!/usr/bin/env python3
"""Update version in package __init__.py file."""

import sys
import re
from pathlib import Path

def update_version(new_version: str) -> None:
    """Update version in holographic_memory/__init__.py"""
    init_file = Path("holographic_memory/__init__.py")
    
    if not init_file.exists():
        print(f"Error: {init_file} not found")
        sys.exit(1)
    
    content = init_file.read_text()
    
    # Update version string
    pattern = r'__version__ = "[^"]*"'
    replacement = f'__version__ = "{new_version}"'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("Warning: Version string not found or not updated")
        sys.exit(1)
    
    init_file.write_text(new_content)
    print(f"Updated version to {new_version}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    update_version(sys.argv[1])
