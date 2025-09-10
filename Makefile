# HolographicMemory Development Makefile
# =====================================
# GPU-Accelerated Wave-Based Storage Engine

PY?=python3
PIP?=$(PY) -m pip

.PHONY: setup test build native api soa clean help

# Default target
help:
	@echo "HolographicMemory Development Commands"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup     Set up dependencies and build native extensions"
	@echo "  make soa       Start SOA system with GPU acceleration"
	@echo "  make test      Run all tests"
	@echo ""
	@echo "Development:"
	@echo "  make native    Build C++ extensions with Metal GPU"
	@echo "  make build     Build all components"
	@echo "  make api       Start API server"
	@echo "  make clean     Clean build artifacts"

# Set up dependencies and build
setup:
	$(PIP) install --upgrade pip wheel setuptools
	$(PIP) install -r services/api/requirements.txt
	$(PIP) install -e services/core
	$(MAKE) native

# Build native C++ extensions with Metal GPU
native:
	@which clang++ > /dev/null || (echo "❌ clang++ not found. Install Xcode Command Line Tools: xcode-select --install" && exit 1)
	@which cmake > /dev/null || (echo "❌ cmake not found. Install with: brew install cmake" && exit 1)
	@echo "🔧 Building GPU backend with CMake..."
	@cd services/core/native/holographic && \
		mkdir -p build && \
		cd build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_METAL=ON && \
		make -j$(shell sysctl -n hw.ncpu)
	@echo "✅ Native extensions built successfully"

# Build all components
build: native
	@echo "🔨 Building all components..."
	@cd services/core && $(PY) -m build
	@echo "✅ All components built"

# Install core service
install:
	@echo "📦 Installing core service..."
	@$(PIP) install -e services/core
	@echo "✅ Core service installed"

# Run all tests
test:
	@echo "🧪 Running all tests..."
	@cd services/core && $(PY) -m pytest -v
	@cd services/api && $(PY) -m pytest -v
	@echo "✅ All tests passed"

# Test GPU acceleration
test-gpu:
	@echo "🌌 Testing GPU acceleration..."
	@$(PY) -c "import sys; sys.path.append('services/core/native/holographic'); import holographic_gpu; print('GPU platforms:', holographic_gpu.available_platforms())"
	@echo "✅ GPU acceleration working"

# Start API server
api:
	@echo "🚀 Starting API server..."
	@$(PY) -m uvicorn services.api.app:app --reload --port 8000 --host 0.0.0.0

# Start SOA system
soa:
	@echo "🚀 Starting HolographicMemory SOA system..."
	@echo "🏗️  GPU-First Architecture with Service Orchestration"
	@echo "🌐 Open: http://localhost:8000"
	@echo ""
	@$(PY) main.py

# Generate OpenAPI documentation
openapi:
	@echo "📚 Generating OpenAPI documentation..."
	@$(PY) -c "import json; from services.api.app_soa import app; print(json.dumps(app.openapi(), indent=2))" > openapi.json
	@echo "✅ OpenAPI documentation generated"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@$(PY) -m pdoc --html services/core/holographicfs --output-dir docs/api
	@echo "✅ Documentation generated"

# Clean build artifacts and cache
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf **/__pycache__ build dist .pytest_cache
	@rm -rf services/core/*.egg-info
	@rm -rf services/core/native/holographic/build
	@rm -rf services/core/native/holographic/*.so
	@echo "✅ Cleanup completed"

# Show current configuration
config:
	@echo "Current Configuration:"
	@echo "  Python: $(PY)"
	@echo "  Pip: $(PIP)"
	@echo "  Architecture: $(shell uname -m)"
	@echo "  OS: $(shell uname -s)"
	@echo "  GPU: Metal (Apple Silicon)"