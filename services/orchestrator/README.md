# Orchestrator Service

Central service orchestration and coordination for the HolographicMemory SOA system.

## Responsibilities

- **Service Discovery**: Locate and register available services
- **Load Balancing**: Distribute requests across service instances
- **Health Monitoring**: Track service health and availability
- **Request Routing**: Route requests to appropriate services
- **Configuration Management**: Centralized configuration for all services

## Architecture

- **Service Registry**: Maintains registry of available services
- **Health Checker**: Monitors service health and availability
- **Load Balancer**: Distributes load across service instances
- **Configuration Store**: Centralized configuration management

## API Endpoints

- `GET /services` - List all registered services
- `POST /services/register` - Register a new service
- `GET /services/{service_id}/health` - Check service health
- `GET /config` - Get configuration for services

## Configuration

The orchestrator manages configuration for:
- Service endpoints and ports
- Health check intervals
- Load balancing strategies
- Authentication and security settings

## Quick Start

```bash
# Start orchestrator
python orchestrator.py

# Register a service
curl -X POST http://localhost:8000/services/register \
  -H "Content-Type: application/json" \
  -d '{"name": "holographic-memory", "endpoint": "http://localhost:8080"}'
```
