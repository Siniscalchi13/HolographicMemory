Deployment Guide
================

Requirements
- Docker 24+ and Docker Compose v2
- CPU with SSE4.2/AVX2 (x86_64) or ARM64 (Apple M/Graviton)

Configuration (env)
- HOLO_ROOT: data directory inside container (default: /data)
- GRID_SIZE: holographic grid dimension (default: 64)
- HOLO_API_KEY: API key for auth (optional)
- HOLO_ALLOWED_ORIGINS: CORS origins (default desktop dev origins)
- PORT: container listen port (default: 8000)

Run (Development)
1) make build-images
2) make dev
3) API: http://localhost:8000/ (docs at /docs, health at /healthz, metrics at /metrics)

Run (Production)
1) export HOLO_API_KEY=…
2) make prod
3) Attach logs: make logs

Security Hardening
- Non‑root user (UID 10001)
- Minimal base image (python:3.12‑slim)
- Healthcheck and CORS
- Secrets via environment or Docker secrets (preferred in production)

Scaling
- Scale API replicas: docker compose -f docker-compose.prod.yml up --scale api=3 -d
- Place behind a reverse proxy/load balancer (NGINX/Traefik) for TLS and sticky sessions if desired

Kubernetes (example)
- See deploy/k8s/api-deployment.yaml and service.yaml
- For production, package as a Helm chart and inject HOLO_API_KEY via secrets

Observability
- Metrics: /metrics (Prometheus format)
- Health: /healthz
- Logs: uvicorn access logs + container logs (make logs)

Backups
- Persist /data volume (holographic state + index JSON)

