# Docker Configuration Directory

## Overview

The `/deployment/docker/` directory contains all Docker-related configuration files for containerizing the HolographicMemory SOA system. This directory provides enterprise-grade containerization with multi-stage builds, health checks, and production-ready configurations.

## Directory Structure

```
docker/
├── README.md              # This comprehensive guide
├── docker-compose.yml     # Multi-service Docker orchestration
├── Dockerfile             # Multi-stage container build
└── nginx.conf             # Nginx reverse proxy configuration
```

## File Details

### `docker-compose.yml`
**Purpose**: Docker Compose orchestration for multi-container deployment
**Technical Details**:
- **Version**: 3.8 (Docker Compose specification)
- **Services**: 3 containerized services with inter-service communication
- **Networking**: Default bridge network with automatic service discovery
- **Volumes**: Named volumes for data persistence and configuration
- **Health Checks**: HTTP-based health monitoring for all services
- **Restart Policies**: Production-ready restart configurations

**Service Configuration**:

#### `holographic-memory` Service
- **Build Context**: Current directory (`.`)
- **Ports Exposed**:
  - `8081:8081` - Main API endpoint
  - `8082:8082` - Telemetry and metrics
  - `9090:9090` - Monitoring and observability
- **Environment Variables**:
  - `HOLOGRAPHIC_ENV=production` - Production environment flag
  - `GPU_ENABLED=true` - GPU acceleration enabled
  - `LOG_LEVEL=INFO` - Logging verbosity level
- **Volume Mounts**:
  - `./data:/app/data` - Data persistence
  - `./logs:/app/logs` - Log file persistence
- **Restart Policy**: `unless-stopped` (automatic restart on failure)
- **Health Check**:
  - **Test**: `["CMD", "curl", "-f", "http://localhost:8081/health"]`
  - **Interval**: 30 seconds
  - **Timeout**: 10 seconds
  - **Retries**: 3 attempts
  - **Start Period**: 40 seconds (grace period for startup)

#### `nginx` Service
- **Image**: `nginx:alpine` (lightweight, secure base image)
- **Ports Exposed**:
  - `80:80` - HTTP traffic
  - `443:443` - HTTPS traffic
- **Volume Mounts**:
  - `./nginx.conf:/etc/nginx/nginx.conf` - Custom Nginx configuration
  - `./ssl:/etc/nginx/ssl` - SSL certificates and keys
- **Dependencies**: `holographic-memory` (waits for main service)
- **Restart Policy**: `unless-stopped`

#### `monitoring` Service
- **Image**: `prom/prometheus:latest` (metrics collection)
- **Ports Exposed**:
  - `9091:9090` - Prometheus web interface (external port 9091)
- **Volume Mounts**:
  - `./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml` - Prometheus config
- **Dependencies**: `holographic-memory` (monitors main service)
- **Restart Policy**: `unless-stopped`

### `Dockerfile`
**Purpose**: Multi-stage Docker build for optimized container image
**Technical Details**:

#### Base Image and System Setup
- **Base Image**: `python:3.13-slim` (minimal Python 3.13 runtime)
- **System Dependencies**:
  - `build-essential` - C/C++ compiler toolchain
  - `cmake` - Cross-platform build system
  - `git` - Version control system
- **Cleanup**: `rm -rf /var/lib/apt/lists/*` (reduces image size)

#### Working Directory
- **Working Directory**: `/app` (application root)

#### Python Dependencies
- **Requirements Files**:
  - `requirements.txt` - Main project dependencies
  - `services/holographic-memory/api/requirements.txt` - API-specific dependencies
- **Installation**: `pip install --no-cache-dir` (optimized, no cache)

#### Source Code and Build
- **Source Copy**: `COPY . .` (copies entire project)
- **GPU Build Process**:
  ```bash
  cd services/holographic-memory/core/native/holographic/build && \
  make clean && make || echo "GPU build failed, continuing with CPU-only"
  ```
  - Attempts GPU component compilation
  - Falls back to CPU-only if GPU build fails
  - Continues deployment regardless of GPU build status

#### Port Configuration
- **Exposed Ports**:
  - `8081` - Main API service
  - `8082` - Telemetry service
  - `9090` - Monitoring service

#### Health Check
- **Type**: HTTP GET request
- **Endpoint**: `http://localhost:8081/health`
- **Configuration**:
  - **Interval**: 30 seconds
  - **Timeout**: 10 seconds
  - **Start Period**: 5 seconds (initial grace period)
  - **Retries**: 3 attempts before marking unhealthy

#### Startup Command
- **Command**: `["python3.13", "main.py"]`
- **Entry Point**: Main application entry point

### `nginx.conf`
**Purpose**: Nginx reverse proxy configuration with SSL termination
**Technical Details**:

#### Events Block
- **Worker Connections**: 1024 (concurrent connections per worker)

#### HTTP Block
- **Upstream Configuration**:
  - **Name**: `holographic_memory`
  - **Backend**: `server holographic-memory:8081`
  - **Load Balancing**: Ready for multiple backend instances

#### HTTP Server (Port 80)
- **Listen**: Port 80
- **Server Name**: `localhost`
- **Redirect**: `return 301 https://$server_name$request_uri`
- **Purpose**: Automatic HTTPS redirect for security

#### HTTPS Server (Port 443)
- **Listen**: Port 443 with SSL
- **Server Name**: `localhost`
- **SSL Configuration**:
  - **Certificate**: `/etc/nginx/ssl/cert.pem`
  - **Private Key**: `/etc/nginx/ssl/key.pem`

#### Proxy Configuration
- **Location**: `/` (all requests)
- **Proxy Pass**: `http://holographic_memory` (upstream backend)
- **Headers**:
  - `Host $host` - Preserves original host header
  - `X-Real-IP $remote_addr` - Client IP address
  - `X-Forwarded-For $proxy_add_x_forwarded_for` - Forwarded IP chain
  - `X-Forwarded-Proto $scheme` - Original protocol (HTTP/HTTPS)

## Deployment Instructions

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 10GB+ free disk space

### Quick Start
```bash
# Navigate to deployment directory
cd deployment/docker

# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f holographic-memory

# Health check
curl http://localhost:8081/health
```

### Production Deployment
```bash
# Build production images
docker-compose build --no-cache

# Start with production configuration
docker-compose -f docker-compose.yml up -d

# Verify all services are healthy
docker-compose ps
docker-compose logs holographic-memory
```

### SSL Certificate Setup
```bash
# Generate self-signed certificates (development)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes

# Set proper permissions
chmod 600 ssl/key.pem ssl/cert.pem

# For production, use proper CA-signed certificates
```

## Environment Variables

### Core Application
- `HOLOGRAPHIC_ENV`: Environment type (development/production)
- `GPU_ENABLED`: Enable GPU acceleration (true/false)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

### Service Configuration
- `HOLO_ROOT`: Data root directory path
- `GRID_SIZE`: Holographic grid size configuration
- `PORT`: Service port number
- `HOLO_API_KEY`: API authentication key

## Volume Management

### Data Persistence
- **Data Volume**: `./data:/app/data`
  - Stores holographic memory patterns
  - Persists across container restarts
  - Backup recommended for production

### Log Persistence
- **Logs Volume**: `./logs:/app/logs`
  - Application logs
  - System logs
  - Monitoring data

### SSL Certificates
- **SSL Volume**: `./ssl:/etc/nginx/ssl`
  - SSL certificates
  - Private keys
  - Certificate chains

## Monitoring and Health Checks

### Health Check Endpoints
- **Application Health**: `http://localhost:8081/health`
- **Prometheus Metrics**: `http://localhost:9091`
- **Nginx Status**: Built-in Nginx status module

### Log Monitoring
```bash
# View real-time logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f holographic-memory
docker-compose logs -f nginx
docker-compose logs -f monitoring

# View logs with timestamps
docker-compose logs -f -t
```

### Performance Monitoring
```bash
# Container resource usage
docker stats

# Service health status
docker-compose ps

# Detailed service information
docker-compose config
```

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check for port conflicts
netstat -tulpn | grep :8081
netstat -tulpn | grep :80
netstat -tulpn | grep :443

# Use different ports if needed
# Edit docker-compose.yml and change port mappings
```

#### Container Startup Issues
```bash
# Check container logs
docker-compose logs holographic-memory

# Check container status
docker-compose ps

# Restart specific service
docker-compose restart holographic-memory
```

#### SSL Certificate Issues
```bash
# Verify certificate files exist
ls -la ssl/

# Check certificate validity
openssl x509 -in ssl/cert.pem -text -noout

# Regenerate certificates if needed
rm ssl/cert.pem ssl/key.pem
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
```

#### GPU Issues
```bash
# Check if GPU is available in container
docker-compose exec holographic-memory nvidia-smi

# Check GPU build logs
docker-compose logs holographic-memory | grep -i gpu

# Force CPU-only mode
# Set GPU_ENABLED=false in docker-compose.yml
```

### Performance Optimization

#### Resource Limits
```yaml
# Add to docker-compose.yml
services:
  holographic-memory:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

#### Scaling
```bash
# Scale holographic-memory service
docker-compose up -d --scale holographic-memory=3

# Update nginx upstream for load balancing
# Edit nginx.conf to include multiple backend servers
```

## Security Considerations

### Container Security
- **Non-root User**: Application runs as non-root user
- **Read-only Filesystem**: Where possible, use read-only filesystems
- **Security Scanning**: Regular vulnerability scanning of images
- **Image Updates**: Keep base images updated

### Network Security
- **Internal Communication**: Services communicate via internal Docker network
- **SSL Termination**: All external traffic encrypted via Nginx
- **Firewall**: Configure host firewall for exposed ports only

### Data Security
- **Volume Encryption**: Consider encrypted volumes for sensitive data
- **Backup Encryption**: Encrypt backups of persistent data
- **Access Control**: Limit access to Docker daemon and volumes

## Maintenance

### Regular Tasks
- **Image Updates**: Regular updates of base images
- **Security Patches**: Apply security updates promptly
- **Log Rotation**: Implement log rotation to prevent disk space issues
- **Backup**: Regular backups of persistent volumes

### Monitoring
- **Health Checks**: Monitor health check endpoints
- **Resource Usage**: Monitor CPU, memory, and disk usage
- **Log Analysis**: Regular analysis of application logs
- **Performance Metrics**: Monitor response times and throughput

## Support

### Documentation
- **Docker Documentation**: [Docker Official Docs](https://docs.docker.com/)
- **Docker Compose**: [Compose Documentation](https://docs.docker.com/compose/)
- **Nginx Configuration**: [Nginx Documentation](http://nginx.org/en/docs/)

### Troubleshooting Resources
- **Container Logs**: Primary source for debugging
- **Health Check Endpoints**: Service status monitoring
- **Docker Commands**: Standard Docker troubleshooting commands
- **Application Logs**: Detailed application-specific logging

## Conclusion

The Docker configuration directory provides a complete containerization solution for the HolographicMemory SOA system. With multi-service orchestration, health monitoring, SSL termination, and production-ready configurations, this setup enables reliable, scalable, and secure deployment of the holographic memory system in any containerized environment.

The configuration supports both development and production use cases, with automatic failover, health monitoring, and comprehensive logging for enterprise-grade reliability.
