#!/usr/bin/env python3
"""
Enhanced File API for Holographic Memory System
Provides advanced file processing, analysis, and holographic storage capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
import asyncio
from pathlib import Path

from .file_processor import file_processor
from .app import require_api_key, get_fs, counter_store

logger = logging.getLogger(__name__)

# Create router for enhanced file operations
router = APIRouter(prefix="/api/v2/files", tags=["Enhanced File Processing"])


@router.post("/upload")
async def upload_file_enhanced(
    file: UploadFile = File(...),
    extract_content: bool = Query(True, description="Extract and analyze file content"),
    generate_thumbnail: bool = Query(True, description="Generate thumbnail for supported formats"),
    store_holographic: bool = Query(True, description="Store in holographic memory"),
    _: bool = Depends(require_api_key)
):
    """
    Enhanced file upload with comprehensive processing and analysis.
    
    Features:
    - Automatic content extraction (PDF, DOCX, TXT, CSV, etc.)
    - Metadata analysis and language detection
    - Thumbnail generation for images and PDFs
    - Holographic memory storage with optimized routing
    - Content-based layer assignment
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    try:
        # Read file content
        content = await file.read()
        logger.info(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        # Get comprehensive file information
        file_info = file_processor.get_file_info(file.filename, content)
        
        # Store in holographic memory if requested
        doc_id = None
        storage_info = {}
        
        if store_holographic:
            try:
                # Use existing storage logic from app.py
                fs = get_fs()
                counter_store.inc()
                
                # Get routing decision
                metadata = {
                    "filename": file.filename,
                    "content_type": file.content_type or '',
                    "file_info": file_info
                }
                
                # Enhanced routing based on file analysis
                routing = _get_enhanced_routing(content, file_info, metadata)
                
                # Generate document ID
                if routing.get("vault"):
                    doc_id = _generate_vault_id()
                else:
                    doc_id = _generate_content_hash(content)
                
                # Store in holographic memory
                storage_result = await _store_in_holographic_memory(
                    fs, content, doc_id, routing, file_info
                )
                
                storage_info = {
                    "doc_id": doc_id,
                    "routing": routing,
                    "storage_result": storage_result
                }
                
            except Exception as e:
                logger.error(f"Error storing in holographic memory: {e}")
                storage_info = {"error": str(e)}
        
        # Prepare response
        response = {
            "success": True,
            "file_info": file_info,
            "storage": storage_info,
            "processing_stats": file_processor.get_processing_stats()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@router.post("/batch-upload")
async def batch_upload_files(
    files: List[UploadFile] = File(...),
    extract_content: bool = Query(True),
    generate_thumbnails: bool = Query(True),
    store_holographic: bool = Query(True),
    _: bool = Depends(require_api_key)
):
    """
    Batch upload multiple files with parallel processing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files (max 50)")
    
    results = []
    errors = []
    
    # Process files in parallel
    tasks = []
    for file in files:
        task = _process_single_file(
            file, extract_content, generate_thumbnails, store_holographic
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Separate results and errors
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append({
                "filename": files[i].filename,
                "error": str(result)
            })
        else:
            successful_results.append(result)
    
    return JSONResponse(content={
        "success": len(errors) == 0,
        "processed": len(successful_results),
        "errors": len(errors),
        "results": successful_results,
        "error_details": errors
    })


@router.get("/supported-formats")
async def get_supported_formats(_: bool = Depends(require_api_key)):
    """Get list of supported file formats and processing capabilities."""
    return JSONResponse(content={
        "supported_formats": file_processor.get_processing_stats(),
        "max_file_size": "100MB",
        "batch_limit": 50
    })


@router.post("/analyze")
async def analyze_file_content(
    file: UploadFile = File(...),
    analysis_type: str = Query("full", description="Type of analysis: full, metadata, content, thumbnail"),
    _: bool = Depends(require_api_key)
):
    """
    Analyze file content without storing in holographic memory.
    Useful for preview and analysis before storage.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    try:
        content = await file.read()
        file_info = file_processor.get_file_info(file.filename, content)
        
        # Filter response based on analysis type
        if analysis_type == "metadata":
            response = {
                "filename": file_info["filename"],
                "metadata": file_info["metadata"],
                "size": file_info["size"],
                "pages": file_info["pages"],
                "word_count": file_info["word_count"]
            }
        elif analysis_type == "content":
            response = {
                "filename": file_info["filename"],
                "text_content": file_info["text_content"][:1000] + "..." if len(file_info["text_content"]) > 1000 else file_info["text_content"],
                "word_count": file_info["word_count"],
                "language": file_info["language"]
            }
        elif analysis_type == "thumbnail":
            response = {
                "filename": file_info["filename"],
                "thumbnail": file_info["thumbnail"],
                "content_type": file_info["content_type"]
            }
        else:  # full
            response = file_info
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error analyzing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")


@router.get("/processing-stats")
async def get_processing_statistics(_: bool = Depends(require_api_key)):
    """Get file processing statistics and system capabilities."""
    return JSONResponse(content={
        "processor_stats": file_processor.get_processing_stats(),
        "system_info": {
            "max_concurrent_uploads": 10,
            "supported_encodings": ["utf-8", "latin-1", "cp1252", "iso-8859-1"],
            "thumbnail_sizes": [200, 400, 800],
            "content_extraction_limits": {
                "pdf_pages": 10,
                "csv_rows": 1000,
                "text_preview": 1000
            }
        }
    })


# Helper functions

async def _process_single_file(
    file: UploadFile,
    extract_content: bool,
    generate_thumbnails: bool,
    store_holographic: bool
) -> Dict[str, Any]:
    """Process a single file asynchronously."""
    try:
        content = await file.read()
        file_info = file_processor.get_file_info(file.filename, content)
        
        result = {
            "filename": file.filename,
            "file_info": file_info,
            "success": True
        }
        
        if store_holographic:
            # Store in holographic memory
            fs = get_fs()
            routing = _get_enhanced_routing(content, file_info, {"filename": file.filename})
            doc_id = _generate_content_hash(content)
            
            storage_result = await _store_in_holographic_memory(
                fs, content, doc_id, routing, file_info
            )
            
            result["storage"] = {
                "doc_id": doc_id,
                "routing": routing,
                "result": storage_result
            }
        
        return result
        
    except Exception as e:
        return {
            "filename": file.filename,
            "success": False,
            "error": str(e)
        }


def _get_enhanced_routing(content: bytes, file_info: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced routing based on file analysis."""
    # Base routing
    routing = {
        "vault": False,
        "format": "v4",
        "layers": [("knowledge", 1.0)],
        "K": 32
    }
    
    # Enhanced routing based on file analysis
    content_type = file_info.get("content_type", "unknown")
    word_count = file_info.get("word_count", 0)
    pages = file_info.get("pages", 0)
    language = file_info.get("language", "unknown")
    
    # Adjust format based on content size
    if word_count < 100:
        routing["format"] = "micro"
    elif word_count < 1000:
        routing["format"] = "microk8"
    else:
        routing["format"] = "v4"
    
    # Adjust layers based on content type
    if content_type == "document":
        routing["layers"] = [("knowledge", 0.8), ("context", 0.2)]
    elif content_type == "data":
        routing["layers"] = [("experience", 0.6), ("knowledge", 0.4)]
    elif content_type == "image":
        routing["layers"] = [("context", 1.0)]
    
    # Adjust K based on content complexity
    if word_count > 10000:
        routing["K"] = 64
    elif word_count > 1000:
        routing["K"] = 32
    else:
        routing["K"] = 16
    
    return routing


def _generate_vault_id() -> str:
    """Generate a secure vault ID."""
    import secrets
    return secrets.token_hex(16)


def _generate_content_hash(content: bytes) -> str:
    """Generate content-based hash for document ID."""
    import hashlib
    return hashlib.sha256(content).hexdigest()


async def _store_in_holographic_memory(
    fs, content: bytes, doc_id: str, routing: Dict[str, Any], file_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Store content in holographic memory system."""
    try:
        # Use existing storage logic from app.py
        # This is a simplified version - in practice, you'd integrate with the existing storage system
        
        # Create patterns directory
        patterns_dir = Path("data/holographic_memory/patterns")
        patterns_dir.mkdir(parents=True, exist_ok=True)
        
        # Store holographic pattern
        stem = Path(file_info["filename"]).stem
        hwp_path = patterns_dir / f"{stem}.hwp"
        
        # For now, store basic file info as JSON
        import json
        pattern_data = {
            "doc_id": doc_id,
            "file_info": file_info,
            "routing": routing,
            "content_hash": _generate_content_hash(content)
        }
        
        with open(hwp_path, 'w') as f:
            json.dump(pattern_data, f, indent=2)
        
        return {
            "success": True,
            "pattern_path": str(hwp_path),
            "storage_size": len(json.dumps(pattern_data))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
