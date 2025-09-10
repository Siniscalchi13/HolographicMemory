# HolographicMemory SOA API Documentation

**Version:** 1.0.0  
**Generated:** 2025-09-10T16:48:32.460739  
**Base URL:** http://localhost:8081

## Overview

Comprehensive API documentation for the HolographicMemory Service-Oriented Architecture

## Authentication

API uses Bearer token authentication

**Header:** `Authorization: Bearer <token>`

## Rate Limiting

- **Requests per minute:** 100
- **Burst limit:** 200
- **Description:** Rate limiting is applied per IP address

## API Endpoints

### Holographic Memory

**Base Path:** `/api/v1/holographic-memory`

Core holographic memory operations

#### POST /store

Store data in holographic memory

**Parameters:**

- `data` (string) ✓ - Data to store
- `metadata` (object) ○ - Optional metadata

**Responses:**

- `200` - Data stored successfully
- `400` - Invalid input data
- `500` - Internal server error

#### GET /retrieve

Retrieve data from holographic memory

**Parameters:**

- `id` (string) ✓ - Memory ID

**Responses:**

- `200` - Data retrieved successfully
- `404` - Data not found
- `500` - Internal server error

#### POST /search

Search holographic memory

**Parameters:**

- `query` (string) ✓ - Search query
- `similarity_threshold` (float) ○ - Similarity threshold

**Responses:**

- `200` - Search results
- `400` - Invalid query
- `500` - Internal server error

### File Processing

**Base Path:** `/api/v1/files`

File processing operations

#### POST /upload

Upload and process files

**Parameters:**

- `file` (file) ✓ - File to upload
- `process_type` (string) ○ - Processing type

**Responses:**

- `200` - File processed successfully
- `400` - Invalid file
- `500` - Processing error

#### POST /process

Process uploaded file

**Parameters:**

- `file_id` (string) ✓ - File ID
- `options` (object) ○ - Processing options

**Responses:**

- `200` - Processing completed
- `404` - File not found
- `500` - Processing error

### Compression

**Base Path:** `/api/v1/compression`

Compression pipeline operations

#### POST /compress

Compress data using holographic compression

**Parameters:**

- `data` (string) ✓ - Data to compress
- `algorithm` (string) ○ - Compression algorithm
- `threshold` (float) ○ - Compression threshold

**Responses:**

- `200` - Data compressed successfully
- `400` - Invalid data
- `500` - Compression error

#### POST /decompress

Decompress holographic data

**Parameters:**

- `compressed_data` (string) ✓ - Compressed data
- `algorithm` (string) ○ - Decompression algorithm

**Responses:**

- `200` - Data decompressed successfully
- `400` - Invalid compressed data
- `500` - Decompression error

### Monitoring

**Base Path:** `/api/v1/monitoring`

System monitoring and health checks

#### GET /health

Get system health status

**Responses:**

- `200` - Health status
- `503` - Service unavailable

#### GET /metrics

Get system metrics

**Parameters:**

- `time_range` (string) ○ - Time range for metrics

**Responses:**

- `200` - System metrics
- `500` - Metrics unavailable

#### GET /alerts

Get active alerts

**Responses:**

- `200` - Active alerts
- `500` - Alerts unavailable

## Error Codes

- `400` - Bad Request - Invalid input parameters
- `401` - Unauthorized - Invalid or missing authentication
- `403` - Forbidden - Insufficient permissions
- `404` - Not Found - Resource not found
- `429` - Too Many Requests - Rate limit exceeded
- `500` - Internal Server Error - Server error
- `503` - Service Unavailable - Service temporarily unavailable

## Examples

### Store Data

```bash
curl -X POST http://localhost:8081/api/v1/holographic-memory/store \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"data": "Hello, Holographic Memory!", "metadata": {"type": "text"}}'
```

### Search Memory

```bash
curl -X POST http://localhost:8081/api/v1/holographic-memory/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "holographic", "similarity_threshold": 0.8}'
```

## SDKs and Tools

- **OpenAPI Specification:** [openapi_spec.json](openapi_spec.json)
- **JSON Documentation:** [api_documentation.json](api_documentation.json)
- **Interactive Documentation:** Available at `/docs` endpoint when API is running

