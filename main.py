#!/usr/bin/env python3
"""
HolographicMemory - Main Entry Point
TAI-style service orchestration and startup
"""

import asyncio
import sys
import os
from pathlib import Path

# Add services to Python path
sys.path.insert(0, str(Path(__file__).parent / "services"))

async def start_services():
    """Start all SOA services"""
    print("🌐 Starting HolographicMemory SOA System...")
    
    # Import services
    try:
        from services.orchestrator.orchestrator import Orchestrator
        from services.holographic_memory.api.app_soa import app as api_app
        import uvicorn
        
        # Start orchestrator
        orchestrator = Orchestrator()
        await orchestrator.start()
        print("✅ Orchestrator started")
        
        # Start API service
        print("🚀 Starting API service on port 8080...")
        config = uvicorn.Config(
            api_app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            reload=False
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all services are properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("HolographicMemory SOA System")
        print("Usage: python main.py")
        print("Starts all services in the SOA architecture")
        return
    
    try:
        asyncio.run(start_services())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()