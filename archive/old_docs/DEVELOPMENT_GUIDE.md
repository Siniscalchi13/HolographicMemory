# 🌌 HolographicMemory Development Guide

## 🚀 Quick Start

### Single Command Development
```bash
make dev            # Start development server with Metal GPU
```

### First Time Setup
```bash
make setup          # Set up virtual environment and dependencies
make dev            # Start development server
```

## 🎯 Development Workflows

### 1. Standard Development (Virtual Environment)
```bash
make dev            # Start FastAPI server with hot reload
```
- ✅ Uses virtual environment
- ✅ Hot reload enabled
- ✅ Auto port selection
- ✅ Metal GPU acceleration

### 2. Embedded Runtime Development (Same as Native App)
```bash
make app-dev        # Prepare embedded Python runtime
make dev-embedded   # Start server with embedded Python
```
- ✅ Uses same Python runtime as native app
- ✅ Identical to production app environment
- ✅ Metal GPU acceleration
- ✅ Perfect for testing app components

### 3. Native macOS App Development
```bash
make app-dev        # Prepare native macOS app
# Open HolographicMemory.xcodeproj in Xcode
# Click Run
```
- ✅ One-click launch
- ✅ Status bar app
- ✅ Browser auto-opens
- ✅ Professional distribution ready

### 4. Docker Development
```bash
make docker-dev     # Start Docker development environment
```
- ✅ Cross-platform development
- ✅ CUDA/ROCm support
- ✅ Isolated environment

## 🔧 Available Commands

### Development
- `make dev` - Start development server (virtual environment)
- `make dev-embedded` - Start with embedded Python runtime
- `make dev-venv` - Start in virtual environment
- `make setup` - Set up development environment

### Building
- `make native` - Build C++ extensions with Metal GPU
- `make build` - Build all components
- `make install` - Install holographicfs package

### Testing
- `make test` - Run all tests
- `make test-gpu` - Test GPU acceleration
- `make benchmark` - Run performance benchmarks
- `make validate` - Validate system integration

### Native App
- `make app-dev` - Prepare native macOS app
- `make app-build` - Build native macOS app
- `make app-test` - Test app components

### Docker
- `make docker-dev` - Docker development environment
- `make docker-prod` - Production Docker image
- `make docker-cuda` - CUDA-enabled Docker image

### Utilities
- `make clean` - Clean build artifacts
- `make help` - Show all commands
- `make config` - Show configuration
- `make openapi` - Generate API docs

## ⚙️ Configuration

### Environment Variables
```bash
# Python executable
PY=python3

# Virtual environment directory
VENV_DIR=venv

# Data directory
HOLO_ROOT=./data

# Grid size
GRID_SIZE=64

# Port (auto-selected if not set)
PORT=8000
```

### Port Management
The Makefile automatically selects available ports to avoid conflicts:
```bash
make dev            # Auto-select port
PORT=9000 make dev  # Use specific port
```

## 🎨 Development Features

### Hot Reload
- Code changes automatically reload
- No need to restart server
- Fast development iteration

### Dynamic Port Selection
- Automatically finds available ports
- Avoids conflicts with other projects
- Easy to override with PORT=9000

### Virtual Environment
- Isolated Python environment
- Clean dependency management
- Easy to reset with `make clean`

### GPU Acceleration
- Metal on macOS (451K+ ops/sec)
- CUDA on NVIDIA
- ROCm on AMD
- Automatic platform detection

## 📱 Native macOS App

### Development Workflow
1. **Prepare**: `make app-dev`
2. **Develop**: Open Xcode project
3. **Test**: `make app-test`
4. **Build**: `make app-build`

### Features
- ✅ One-click launch
- ✅ Status bar app
- ✅ Browser auto-opens
- ✅ Metal GPU acceleration
- ✅ App Store ready

### Performance
- **Target**: 451K+ ops/sec
- **Achieved**: 1,029,232 ops/sec
- **Compression**: 9x ratio
- **Storage**: Sub-millisecond

## 🐳 Docker Development

### Development Environment
```bash
make docker-dev     # Start development environment
```
- Linux containers
- CUDA/ROCm support
- Cross-platform testing

### Production Build
```bash
make docker-prod    # Build production image
```
- Optimized for production
- Minimal dependencies
- Security hardened

### CUDA Support
```bash
make docker-cuda    # Build CUDA-enabled image
```
- NVIDIA GPU support
- CUDA runtime
- High-performance computing

## 🧪 Testing & Validation

### Test Commands
```bash
make test           # Run all tests
make test-gpu       # Test GPU acceleration
make benchmark      # Performance benchmarks
make validate       # System validation
```

### Performance Testing
```bash
make benchmark      # Comprehensive benchmarks
make benchmark-gpu  # GPU performance
make benchmark-cpu  # CPU performance
```

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Auto-select available port
make dev

# Or specify different port
PORT=9000 make dev
```

#### Virtual Environment Issues
```bash
# Clean and recreate
make clean
make setup
```

#### GPU Not Working
```bash
# Test GPU
make test-gpu

# Check Metal availability
python3 -c "import Metal; print('Metal available')"
```

#### Native Extensions Not Building
```bash
# Clean and rebuild
make clean
make native
```

### Debug Mode
```bash
# Verbose output
make dev V=1

# Debug Python
PYTHONPATH=. python3 -m uvicorn services.api.app:app --reload --port 8000
```

## 📊 Performance Targets

### Metal GPU (macOS)
- **Target**: 451K+ ops/sec
- **Achieved**: 1,029,232 ops/sec
- **Memory**: 9x compression ratio

### CUDA (NVIDIA)
- **Target**: 200K+ ops/sec
- **Memory**: Pinned host memory
- **Streams**: Async processing

### ROCm (AMD)
- **Target**: 200K+ ops/sec
- **Memory**: HIP memory management
- **Streams**: Async processing

## 🎉 Success Indicators

### Development Server
```
✅ Server running on http://localhost:8000
✅ Metal GPU acceleration active
✅ Hot reload enabled
✅ Virtual environment active
```

### Native App
```
✅ Xcode project generated
✅ Python.framework embedded
✅ Metal GPU module built
✅ FastAPI server bundled
```

### Performance
```
✅ 451K+ ops/sec (Metal)
✅ 9x compression ratio
✅ Sub-millisecond storage
✅ Real-time metrics
```

## 🚀 Ready to Start?

```bash
make setup          # First time setup
make dev            # Start developing
make help           # Show all commands
```

**Happy coding! 🌌✨**
