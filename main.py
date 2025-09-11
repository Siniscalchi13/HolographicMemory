#!/usr/bin/env python3
"""
HolographicMemory - Main Entry Point
TAI-style service orchestration and startup
"""

import asyncio
import sys
import os
import importlib.util
from pathlib import Path

# Add services to Python path
sys.path.insert(0, str(Path(__file__).parent / "services"))
sys.path.insert(0, str(Path(__file__).parent / "services" / "holographic-memory" / "api"))

async def start_services():
    """Start all SOA services"""
    print("🌐 Starting HolographicMemory SOA System...")
    
    # Import services
    try:
        from services.orchestrator.orchestrator import HolographicMemoryOrchestrator
        import uvicorn
        
        # Import app_soa using importlib since the directory has hyphens
        app_soa_path = Path(__file__).parent / "services" / "holographic-memory" / "api" / "app_soa.py"
        spec = importlib.util.spec_from_file_location("app_soa", app_soa_path)
        app_soa_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_soa_module)
        api_app = app_soa_module.app
        
        # Initialize orchestrator
        state_dir = Path(__file__).parent / "data" / "production"
        state_dir.mkdir(parents=True, exist_ok=True)
        orchestrator = HolographicMemoryOrchestrator(state_dir=state_dir)
        print("✅ Orchestrator initialized")
        
        # Get port from command line or use default
        port = 8080
        if len(sys.argv) > 1 and sys.argv[1] == "--port":
            try:
                port = int(sys.argv[2])
            except (IndexError, ValueError):
                print("⚠️  Invalid port, using default 8080")
        
        # Start API service
        print(f"🚀 Starting API service on port {port}...")
        config = uvicorn.Config(
            api_app,
            host="0.0.0.0",
            port=port,
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