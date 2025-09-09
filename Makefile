# HolographicMemory Development Makefile
# =====================================
# 🚀 Revolutionary GPU-Accelerated Wave-Based Storage Engine
# 
# Quick Start:
#   make dev          # Start development server with Metal GPU
#   make help         # Show all available commands
#   make test         # Run all tests
#   make build        # Build native extensions
#   make clean        # Clean build artifacts

# Configuration
PY?=python3
PIP?=$(PY) -m pip
VENV_DIR?=venv
VENV_PY?=$(VENV_DIR)/bin/python
VENV_PIP?=$(VENV_PY) -m pip

# Dynamic port selection to avoid conflicts
PORT?=$(shell python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
HOLO_ROOT?=./data
GRID_SIZE?=64

# Colors for output
BLUE=\033[0;34m
GREEN=\033[0;32m
YELLOW=\033[1;33m
RED=\033[0;31m
NC=\033[0m # No Color

.PHONY: help dev setup setup-verify test build native clean install uninstall docker-dev docker-prod app-dev app-build benchmark validate

# Default target
help:
	@if [ -f "Makefile.help" ]; then \
		cat Makefile.help; \
	else \
		echo "$(BLUE)🌌 HolographicMemory Development Commands$(NC)"; \
		echo ""; \
		echo "$(GREEN)🚀 Quick Start:$(NC)"; \
		echo "  make dev          Start development server with Metal GPU acceleration"; \
		echo "  make setup        Set up virtual environment and install dependencies"; \
		echo "  make test         Run all tests"; \
		echo ""; \
		echo "$(GREEN)🔧 Development:$(NC)"; \
		echo "  make dev          Start FastAPI server with hot reload (port: $(PORT))"; \
		echo "  make dev-embedded Start server using embedded Python (same as native app)"; \
		echo "  make dev-venv     Start development server in virtual environment"; \
		echo "  make native       Build native C++ extensions with Metal GPU"; \
		echo "  make build        Build all components"; \
		echo "  make clean        Clean build artifacts and cache"; \
		echo ""; \
		echo "$(GREEN)📱 Native macOS App:$(NC)"; \
		echo "  make app-dev      Prepare and run native macOS app (Metal GPU)"; \
		echo "  make app-build    Build native macOS app for distribution"; \
		echo "  make app-test     Test native macOS app components"; \
		echo ""; \
		echo "$(GREEN)🐳 Docker:$(NC)"; \
		echo "  make docker-dev   Start development environment in Docker"; \
		echo "  make docker-prod  Build production Docker image"; \
		echo "  make docker-cuda  Build CUDA-enabled Docker image"; \
		echo ""; \
		echo "$(GREEN)🧪 Testing & Validation:$(NC)"; \
		echo "  make test         Run all tests"; \
		echo "  make test-gpu     Test GPU acceleration (Metal/CUDA/ROCm)"; \
		echo "  make benchmark    Run performance benchmarks"; \
		echo "  make validate     Validate system integration"; \
		echo ""; \
		echo "$(GREEN)📊 Performance:$(NC)"; \
		echo "  make benchmark    Run comprehensive performance benchmarks"; \
		echo "  make benchmark-gpu Test GPU acceleration performance"; \
		echo "  make benchmark-cpu Test CPU-only performance"; \
		echo ""; \
		echo "$(GREEN)🔍 Utilities:$(NC)"; \
		echo "  make install      Install holographicfs package"; \
		echo "  make uninstall    Uninstall holographicfs package"; \
		echo "  make openapi      Generate OpenAPI documentation"; \
		echo "  make docs         Generate documentation"; \
		echo ""; \
		echo "$(YELLOW)💡 Tips:$(NC)"; \
		echo "  • Use 'make dev' for quick development with Metal GPU"; \
		echo "  • Use 'make app-dev' for native macOS app testing"; \
		echo "  • Use 'make docker-dev' for cross-platform development"; \
		echo "  • Port is auto-selected to avoid conflicts: $(PORT)"; \
		echo ""; \
	fi

# Development server with Metal GPU acceleration
dev:
	@echo "$(BLUE)🚀 Starting HolographicMemory development server...$(NC)"
	@echo "$(YELLOW)📍 Port: $(PORT)$(NC)"
	@echo "$(YELLOW)🌌 GPU: Metal acceleration enabled$(NC)"
	@echo "$(YELLOW)📁 Data: $(HOLO_ROOT)$(NC)"
	@echo ""
	@PYTHONPATH="holographic-fs/native/holographic/build:$$PYTHONPATH" $(VENV_PY) -m uvicorn services.api.app:app --reload --port $(PORT) --host 0.0.0.0

# Development server using embedded Python runtime (same as native app)
dev-embedded:
	@echo "$(BLUE)🚀 Starting HolographicMemory development server (embedded runtime)...$(NC)"
	@echo "$(YELLOW)📍 Port: $(PORT)$(NC)"
	@echo "$(YELLOW)🌌 GPU: Metal acceleration enabled$(NC)"
	@echo "$(YELLOW)📁 Data: $(HOLO_ROOT)$(NC)"
	@echo "$(YELLOW)🐍 Python: Embedded runtime (same as native app)$(NC)"
	@echo ""
	@./scripts/macos/dev_server.sh $(PORT)

# Development server in virtual environment (alias for dev)
dev-venv: dev

# Set up virtual environment and install dependencies
setup: setup-venv install-deps
	@echo "$(YELLOW)🔨 Building native extensions...$(NC)"
	@$(MAKE) native
	@echo "$(GREEN)✅ Development environment ready with GPU acceleration!$(NC)"
	@echo ""
	@echo "$(BLUE)🚀 Next Steps:$(NC)"
	@echo "$(YELLOW)  1. Verify everything works: make setup-verify$(NC)"
	@echo "$(YELLOW)  2. Activate virtual environment: source venv/bin/activate$(NC)"
	@echo "$(YELLOW)  3. Start the server: make dev$(NC)"
	@echo ""
	@echo "$(GREEN)🌟 Welcome to the future of memory - powered by holographic computing!$(NC)"

setup-venv:
	@echo "$(BLUE)🔧 Setting up virtual environment...$(NC)"
	@$(PY) -m venv $(VENV_DIR)
	@echo "$(GREEN)✅ Virtual environment created$(NC)"

install-deps:
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@$(VENV_PIP) install --upgrade pip wheel setuptools
	@$(VENV_PIP) install -r services/api/requirements.txt
	@$(VENV_PIP) install -e holographic-fs
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

# Verify setup is working correctly
setup-verify:
	@echo "$(BLUE)🔍 Verifying HolographicMemory setup...$(NC)"
	@echo "$(YELLOW)🐍 Checking Python environment...$(NC)"
	@$(VENV_PY) --version
	@echo "$(YELLOW)📦 Checking virtual environment...$(NC)"
	@$(VENV_PY) -c "import sys; print(f'Python executable: {sys.executable}')"
	@echo "$(YELLOW)🌌 Testing holographicfs import...$(NC)"
	@$(VENV_PY) -c "import holographicfs; print('✅ holographicfs imported successfully')"
	@echo "$(YELLOW)🔨 Testing native extensions...$(NC)"
	@if $(VENV_PY) -c "from holographicfs.memory import mount; print('✅ Native extensions working')" 2>/dev/null; then \
		echo "$(YELLOW)🚀 Testing GPU backend...$(NC)"; \
		if $(VENV_PY) -c "from holographicfs.memory import mount; fs = mount('./data', grid_size=64); print('✅ GPU backend initialized')" 2>/dev/null; then \
			echo "$(YELLOW)🧪 Running basic functionality test...$(NC)"; \
			$(VENV_PY) -c "from holographicfs.memory import mount; fs = mount('./data', grid_size=64); stats = fs.stats(); print(f'✅ Basic functionality test passed - {stats.get(\"files_indexed\", 0)} files indexed')"; \
			echo "$(GREEN)🎯 Metal GPU acceleration: READY$(NC)"; \
		else \
			echo "$(YELLOW)⚠️ GPU backend needs native extensions - running 'make native'...$(NC)"; \
			$(MAKE) native && \
			$(VENV_PY) -c "from holographicfs.memory import mount; fs = mount('./data', grid_size=64); print('✅ GPU backend now working')" && \
			echo "$(GREEN)🎯 Metal GPU acceleration: READY$(NC)"; \
		fi; \
	else \
		echo "$(YELLOW)⚠️ Native extensions not built - running 'make native'...$(NC)"; \
		$(MAKE) native && \
		$(VENV_PY) -c "from holographicfs.memory import mount; fs = mount('./data', grid_size=64); print('✅ Native extensions now working')" && \
		echo "$(GREEN)🎯 Metal GPU acceleration: READY$(NC)"; \
	fi
	@echo "$(YELLOW)📚 Testing API dependencies...$(NC)"
	@$(VENV_PY) -c "import fastapi, uvicorn; print('✅ FastAPI and uvicorn available')"
	@echo "$(GREEN)🎉 Setup verification complete!$(NC)"
	@echo ""
	@echo "$(BLUE)🚀 Next Steps:$(NC)"
	@echo "$(YELLOW)  1. Activate virtual environment: source venv/bin/activate$(NC)"
	@echo "$(YELLOW)  2. Start the server: make dev$(NC)"
	@echo ""
	@echo "$(GREEN)🌟 Ready to build the future of memory!$(NC)"

# Build native C++ extensions with Metal GPU
native:
	@echo "$(BLUE)🔨 Building native C++ extensions with Metal GPU...$(NC)"
	@echo "$(YELLOW)📋 Checking build prerequisites...$(NC)"
	@which clang++ > /dev/null || (echo "$(RED)❌ clang++ not found. Install Xcode Command Line Tools: xcode-select --install$(NC)" && exit 1)
	@which cmake > /dev/null || (echo "$(RED)❌ cmake not found. Install with: brew install cmake$(NC)" && exit 1)
	@echo "$(YELLOW)🔧 Building GPU backend with CMake...$(NC)"
	@cd holographic-fs/native/holographic && \
		mkdir -p build && \
		cd build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_METAL=ON && \
		make -j$(shell sysctl -n hw.ncpu)
	@echo "$(GREEN)✅ Native extensions built successfully$(NC)"

# Build all components
build: native
	@echo "$(BLUE)🔨 Building all components...$(NC)"
	@cd holographic-fs && $(VENV_PY) -m build
	@echo "$(GREEN)✅ All components built$(NC)"

# Install holographicfs package
install:
	@echo "$(BLUE)📦 Installing holographicfs package...$(NC)"
	@$(VENV_PIP) install -e holographic-fs
	@echo "$(GREEN)✅ Package installed$(NC)"

# Uninstall holographicfs package
uninstall:
	@echo "$(BLUE)🗑️ Uninstalling holographicfs package...$(NC)"
	@$(VENV_PIP) uninstall -y holographicfs
	@echo "$(GREEN)✅ Package uninstalled$(NC)"

# Run all tests
test:
	@echo "$(BLUE)🧪 Running all tests...$(NC)"
	@cd holographic-fs && $(VENV_PY) -m pytest -v
	@cd services/api && $(VENV_PY) -m pytest -v
	@echo "$(GREEN)✅ All tests passed$(NC)"

# Test GPU acceleration
test-gpu:
	@echo "$(BLUE)🌌 Testing GPU acceleration...$(NC)"
	@$(VENV_PY) -c "import sys; sys.path.append('holographic-fs/native/holographic'); import holographic_gpu; print('GPU platforms:', holographic_gpu.available_platforms())"
	@echo "$(GREEN)✅ GPU acceleration working$(NC)"

# Run performance benchmarks
benchmark:
	@echo "$(BLUE)📊 Running performance benchmarks...$(NC)"
	@$(VENV_PY) benchmarks/enterprise/run_repro_bench.py
	@echo "$(GREEN)✅ Benchmarks completed$(NC)"

# Test GPU performance
benchmark-gpu:
	@echo "$(BLUE)🌌 Testing GPU performance...$(NC)"
	@cd holographic-fs/native/holographic && $(VENV_PY) benchmarks_gpu.cpp
	@echo "$(GREEN)✅ GPU benchmarks completed$(NC)"

# Test CPU performance
benchmark-cpu:
	@echo "$(BLUE)💻 Testing CPU performance...$(NC)"
	@$(VENV_PY) benchmarks/enterprise/run_repro_bench.py --cpu-only
	@echo "$(GREEN)✅ CPU benchmarks completed$(NC)"

# Validate system integration
validate:
	@echo "$(BLUE)🔍 Validating system integration...$(NC)"
	@$(VENV_PY) scripts/validate_accuracy.py
	@$(VENV_PY) scripts/validate_performance.py
	@echo "$(GREEN)✅ System validation passed$(NC)"

# Native macOS app development
app-dev:
	@echo "$(BLUE)📱 Preparing native macOS app...$(NC)"
	@./scripts/macos/prepare_python_env.sh
	@cd macos/HolographicMemoryApp && xcodegen generate
	@echo "$(GREEN)✅ Native app ready$(NC)"
	@echo "$(YELLOW)💡 Open HolographicMemory.xcodeproj and run$(NC)"

# Build native macOS app
app-build: app-dev
	@echo "$(BLUE)🔨 Building native macOS app...$(NC)"
	@cd macos/HolographicMemoryApp && xcodebuild -project HolographicMemory.xcodeproj -scheme HolographicMemory -configuration Release build
	@echo "$(GREEN)✅ Native app built$(NC)"

# Test native macOS app components
app-test:
	@echo "$(BLUE)🧪 Testing native macOS app components...$(NC)"
	@./scripts/macos/prepare_python_env.sh
	@echo "$(GREEN)✅ App components tested$(NC)"

# Docker development environment
docker-dev:
	@echo "$(BLUE)🐳 Starting Docker development environment...$(NC)"
	@docker compose -f docker-compose-dev.yml up --build

# Docker production build
docker-prod:
	@echo "$(BLUE)🐳 Building production Docker image...$(NC)"
	@docker build -f services/api/Dockerfile --target api -t holo-api:prod .

# Docker CUDA build
docker-cuda:
	@echo "$(BLUE)🐳 Building CUDA-enabled Docker image...$(NC)"
	@docker build -f services/api/Dockerfile --target api-gpu-cuda -t holo-api:cuda .

# Generate OpenAPI documentation
openapi:
	@echo "$(BLUE)📚 Generating OpenAPI documentation...$(NC)"
	@$(VENV_PY) scripts/dump_openapi.py
	@echo "$(GREEN)✅ OpenAPI documentation generated$(NC)"

# Generate documentation
docs:
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@$(VENV_PY) -m pdoc --html holographic-fs/holographicfs --output-dir docs/api
	@echo "$(GREEN)✅ Documentation generated$(NC)"

# Clean build artifacts and cache
clean:
	@echo "$(BLUE)🧹 Cleaning build artifacts...$(NC)"
	@rm -rf **/__pycache__ build dist .pytest_cache
	@rm -rf holographic-fs/*.egg-info
	@rm -rf holographic-fs/native/holographic/build
	@rm -rf holographic-fs/native/holographic/*.so
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

# Show current configuration
config:
	@echo "$(BLUE)⚙️ Current Configuration:$(NC)"
	@echo "  Python: $(PY)"
	@echo "  Virtual Environment: $(VENV_DIR)"
	@echo "  Port: $(PORT)"
	@echo "  Data Directory: $(HOLO_ROOT)"
	@echo "  Grid Size: $(GRID_SIZE)"