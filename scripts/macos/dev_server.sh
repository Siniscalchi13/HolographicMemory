#!/bin/bash
# HolographicMemory Development Server
# ====================================
# 🚀 Quick development server using embedded Python runtime
# 
# This script uses the same embedded Python runtime as the native macOS app
# but runs the FastAPI server directly for rapid development iteration.
#
# Usage:
#   ./scripts/macos/dev_server.sh [port]
#
# Examples:
#   ./scripts/macos/dev_server.sh        # Auto-select port
#   ./scripts/macos/dev_server.sh 9000   # Use port 9000

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACOS_DIR="$PROJECT_ROOT/macos/HolographicMemoryApp"
RUNTIME_DIR="$MACOS_DIR/Runtime"

# Port selection
if [ -n "$1" ]; then
    PORT="$1"
else
    # Auto-select available port
    PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi

echo -e "${BLUE}🌌 HolographicMemory Development Server${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if embedded Python runtime exists
if [ ! -d "$RUNTIME_DIR/Python.framework" ]; then
    echo -e "${RED}❌ Embedded Python runtime not found${NC}"
    echo -e "${YELLOW}💡 Run 'make app-dev' first to prepare the runtime${NC}"
    echo ""
    echo "Available commands:"
    echo "  make app-dev    # Prepare embedded Python runtime"
    echo "  make dev        # Use system Python (alternative)"
    exit 1
fi

# Check if GPU module exists
if [ ! -f "$RUNTIME_DIR/Python/site-packages/holographic_gpu.cpython-312-darwin.so" ]; then
    echo -e "${RED}❌ GPU module not found${NC}"
    echo -e "${YELLOW}💡 Run 'make app-dev' first to build the GPU module${NC}"
    exit 1
fi

# Set up environment
export PYTHONPATH="$RUNTIME_DIR/Python/site-packages:$PROJECT_ROOT:$PYTHONPATH"
export DYLD_LIBRARY_PATH="$RUNTIME_DIR/Python.framework/Versions/3.12/lib:$DYLD_LIBRARY_PATH"
export HOLO_ROOT="$PROJECT_ROOT/data"
export GRID_SIZE=64

# Python executable
PYTHON_EXEC="$RUNTIME_DIR/Python.framework/Versions/3.12/bin/python3.12"

echo -e "${GREEN}✅ Environment configured${NC}"
echo -e "${YELLOW}📍 Port: $PORT${NC}"
echo -e "${YELLOW}🌌 GPU: Metal acceleration enabled${NC}"
echo -e "${YELLOW}📁 Data: $HOLO_ROOT${NC}"
echo -e "${YELLOW}🐍 Python: Embedded runtime${NC}"
echo ""

# Test GPU module
echo -e "${BLUE}🔍 Testing GPU module...${NC}"
if $PYTHON_EXEC -c "import holographic_gpu; print('GPU platforms:', holographic_gpu.available_platforms())" 2>/dev/null; then
    echo -e "${GREEN}✅ GPU module working${NC}"
else
    echo -e "${YELLOW}⚠️ GPU module test failed, continuing anyway${NC}"
fi
echo ""

# Create data directory
mkdir -p "$HOLO_ROOT"

# Kill any existing server on this port
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo -e "${YELLOW}🔄 Killing existing server on port $PORT...${NC}"
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start the server
echo -e "${BLUE}🚀 Starting FastAPI server...${NC}"
echo -e "${YELLOW}💡 Press Ctrl+C to stop${NC}"
echo ""

# Change to project root for relative imports
cd "$PROJECT_ROOT"

# Start uvicorn with embedded Python
exec $PYTHON_EXEC -m uvicorn services.api.app:app \
    --reload \
    --port "$PORT" \
    --host 0.0.0.0 \
    --log-level info
