"""
SOA-Based Holographic Memory FastAPI Application

Clean SOA architecture with proper service orchestration.
API service orchestrates, doesn't implement.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.orchestrator.orchestrator import HolographicMemoryOrchestrator
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
store_requests = Counter('holographic_store_requests_total', 'Total store requests')
retrieve_requests = Counter('holographic_retrieve_requests_total', 'Total retrieve requests')
search_requests = Counter('holographic_search_requests_total', 'Total search requests')

# Global orchestrator instance
ORCHESTRATOR: Optional[HolographicMemoryOrchestrator] = None


def get_orchestrator() -> HolographicMemoryOrchestrator:
    """Dependency to get the orchestrator instance."""
    global ORCHESTRATOR
    if ORCHESTRATOR is None:
        data_dir = Path(os.getenv("HOLOGRAPHIC_DATA_DIR", "./data/production"))
        grid_size = int(os.getenv("HOLOGRAPHIC_GRID_SIZE", "1024"))
        use_gpu = os.getenv("HOLOGRAPHIC_USE_GPU", "true").lower() == "true"
        
        ORCHESTRATOR = HolographicMemoryOrchestrator(
            state_dir=data_dir,
            grid_size=grid_size,
            use_gpu=use_gpu
        )
        logger.info(f"Initialized orchestrator with GPU: {use_gpu}, Grid size: {grid_size}")
    
    return ORCHESTRATOR


# FastAPI app
app = FastAPI(
    title="Holographic Memory SOA API",
    description="Service-Oriented Architecture for Holographic Memory System",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="services/holographic-memory/api/static"), name="static")


# Pydantic models
class StoreRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class RebalanceRequest(BaseModel):
    force: bool = False


# Health and status endpoints
@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/capabilities")
async def get_capabilities(orchestrator: HolographicMemoryOrchestrator = Depends(get_orchestrator)):
    """Get system capabilities and status."""
    status = orchestrator.get_system_status()
    return {
        "gpu_available": status["memory_status"]["use_gpu"],
        "backend_type": status["memory_status"]["backend_type"],
        "grid_size": status["memory_status"]["grid_size"],
        "layer_dimensions": status["layer_dimensions"],
        "services": {
            "math_core": status["math_core_status"],
            "router": status["router_status"],
            "vault": status["vault_status"]
        },
        "telemetry": status["telemetry"]
    }


@app.get("/stats")
async def get_stats(orchestrator: HolographicMemoryOrchestrator = Depends(get_orchestrator)):
    """Get system statistics and performance metrics."""
    status = orchestrator.get_system_status()
    return {
        "system_status": status,
        "timestamp": time.time()
    }


# Core storage and retrieval endpoints
@app.post("/store")
async def store_content(
    file: UploadFile = File(...),
    metadata: Optional[str] = Query(None),
    orchestrator: HolographicMemoryOrchestrator = Depends(get_orchestrator)
):
    """Store content using SOA orchestration."""
    store_requests.inc()
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            import json
            parsed_metadata = json.loads(metadata)
        
        # Add file metadata
        parsed_metadata.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content)
        })
        
        # Store using orchestrator
        result = orchestrator.store_content(content, parsed_metadata)
        
        return {
            "success": True,
            "doc_id": result["doc_id"],
            "routing_decision": result["routing_decision"],
            "layer_dimensions": result["layer_dimensions"]
        }
        
    except Exception as e:
        logger.error(f"Store error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{doc_id}")
async def download_content(
    doc_id: str,
    orchestrator: HolographicMemoryOrchestrator = Depends(get_orchestrator)
):
    """Download content by document ID."""
    retrieve_requests.inc()
    
    try:
        result = orchestrator.retrieve_content(doc_id)
        
        if not result["content"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Return as file download
        return StreamingResponse(
            io.BytesIO(result["content"]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={doc_id}"}
        )
        
    except Exception as e:
        logger.error(f"Retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_content(
    request: SearchRequest,
    orchestrator: HolographicMemoryOrchestrator = Depends(get_orchestrator)
):
    """Search content using SOA orchestration."""
    search_requests.inc()
    
    try:
        results = orchestrator.search_content(request.query, request.limit)
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list")
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    orchestrator: HolographicMemoryOrchestrator = Depends(get_orchestrator)
):
    """List stored documents."""
    try:
        # This would need to be implemented in the orchestrator
        # For now, return empty list
        return {
            "success": True,
            "documents": [],
            "count": 0,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System management endpoints
@app.post("/rebalance")
async def rebalance_layers(
    request: RebalanceRequest,
    orchestrator: HolographicMemoryOrchestrator = Depends(get_orchestrator)
):
    """Rebalance layer dimensions using Math Core and Telemetry services."""
    try:
        result = orchestrator.rebalance_layers()
        
        return {
            "success": True,
            "rebalancing_result": result
        }
        
    except Exception as e:
        logger.error(f"Rebalance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Dashboard endpoint
@app.get("/")
async def dashboard():
    """Simple dashboard."""
    return FileResponse("services/holographic-memory/api/static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
