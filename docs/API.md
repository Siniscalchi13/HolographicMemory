API Documentation
=================

Base URL
- Default: http://localhost:8000
- Auth: optional API key via header `X-API-Key` (set HOLO_API_KEY)

Endpoints
- GET /healthz → { status: "ok" }
- GET /stats → store + engine stats
- GET /search?q=…&k=5 → filename index search
- GET /list → list indexed files
- GET /tree → folder tree from index
- POST /store (multipart/form-data) file=… → { doc_id, filename }
- GET /download/{doc_id} → raw bytes (if retrievable)
- GET /thumb?path=…&w=256 → PNG thumbnail or 1x1
- POST /delete { path } → { status: "ok" }
- POST /rename { path, new_path } → { status: "ok" }
- Metrics: GET /metrics (Prometheus)

Errors
- 4xx for user errors with `detail`
- 5xx for server errors with `detail`

Rate Limiting
- Not enforced by default; place behind a gateway/reverse proxy for production limits

Tracing
- Ready for OpenTelemetry via ASGI middleware (add in services/api/app.py when needed)

