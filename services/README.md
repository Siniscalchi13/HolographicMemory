# HolographicMemory Services

This directory contains all the SOA (Service-Oriented Architecture) services for the HolographicMemory system.

## Service Architecture

### Core Services

- **`holographic-memory/`** - Main holographic memory engine with GPU acceleration
- **`orchestrator/`** - Central service orchestration and coordination
- **`math-core/`** - Mathematical computation services
- **`router/`** - Request routing and load balancing
- **`telemetry/`** - Monitoring and observability
- **`vault/`** - Secure data storage and retrieval

### Service Communication

All services communicate through:
- REST APIs for external communication
- Internal message queues for service-to-service communication
- Shared data contracts and schemas

### Development

Each service has its own:
- `requirements.txt` - Python dependencies
- `README.md` - Service-specific documentation
- `tests/` - Unit and integration tests
- `Dockerfile` - Container configuration

### Quick Start

```bash
# Set up all services
make setup

# Start individual services
make api      # Start API service
make soa      # Start full SOA system

# Run tests
make test
```
