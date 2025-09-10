#!/usr/bin/env python3
"""
Holographic Memory System - SOA Entry Point

Main entry point for the SOA-based Holographic Memory system.
Integrates GPU-first holographic-fs with proper service orchestration.
"""

import sys
import os
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent / "services"))

from services.api.app_soa import app, get_orchestrator
import uvicorn


def main():
    """Main entry point for the SOA system."""
    print("🚀 Starting Holographic Memory SOA System")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    status = orchestrator.get_system_status()
    
    print(f"✅ GPU Backend: {status['memory_status']['backend_type']}")
    print(f"✅ GPU Enabled: {status['memory_status']['use_gpu']}")
    print(f"✅ Grid Size: {status['memory_status']['grid_size']}")
    print(f"✅ Services: Math Core, Router, Vault, Telemetry")
    print("=" * 50)
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
