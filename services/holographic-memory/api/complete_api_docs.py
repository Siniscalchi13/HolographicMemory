#!/usr/bin/env python3
"""
API Documentation Completion Script
==================================

This script generates comprehensive API documentation for the SOA system.
"""

import sys
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# Add services to path
sys.path.insert(0, 'services')

def generate_api_documentation():
    """Generate comprehensive API documentation"""
    
    api_docs = {
        "title": "HolographicMemory SOA API Documentation",
        "version": "1.0.0",
        "description": "Comprehensive API documentation for the HolographicMemory Service-Oriented Architecture",
        "generated_at": datetime.now().isoformat(),
        "base_url": "http://localhost:8081",
        "endpoints": {
            "holographic_memory": {
                "base_path": "/api/v1/holographic-memory",
                "description": "Core holographic memory operations",
                "endpoints": [
                    {
                        "path": "/store",
                        "method": "POST",
                        "description": "Store data in holographic memory",
                        "parameters": {
                            "data": {"type": "string", "required": True, "description": "Data to store"},
                            "metadata": {"type": "object", "required": False, "description": "Optional metadata"}
                        },
                        "responses": {
                            "200": {"description": "Data stored successfully", "schema": {"type": "object"}},
                            "400": {"description": "Invalid input data"},
                            "500": {"description": "Internal server error"}
                        }
                    },
                    {
                        "path": "/retrieve",
                        "method": "GET",
                        "description": "Retrieve data from holographic memory",
                        "parameters": {
                            "id": {"type": "string", "required": True, "description": "Memory ID"}
                        },
                        "responses": {
                            "200": {"description": "Data retrieved successfully"},
                            "404": {"description": "Data not found"},
                            "500": {"description": "Internal server error"}
                        }
                    },
                    {
                        "path": "/search",
                        "method": "POST",
                        "description": "Search holographic memory",
                        "parameters": {
                            "query": {"type": "string", "required": True, "description": "Search query"},
                            "similarity_threshold": {"type": "float", "required": False, "description": "Similarity threshold"}
                        },
                        "responses": {
                            "200": {"description": "Search results"},
                            "400": {"description": "Invalid query"},
                            "500": {"description": "Internal server error"}
                        }
                    }
                ]
            },
            "file_processing": {
                "base_path": "/api/v1/files",
                "description": "File processing operations",
                "endpoints": [
                    {
                        "path": "/upload",
                        "method": "POST",
                        "description": "Upload and process files",
                        "parameters": {
                            "file": {"type": "file", "required": True, "description": "File to upload"},
                            "process_type": {"type": "string", "required": False, "description": "Processing type"}
                        },
                        "responses": {
                            "200": {"description": "File processed successfully"},
                            "400": {"description": "Invalid file"},
                            "500": {"description": "Processing error"}
                        }
                    },
                    {
                        "path": "/process",
                        "method": "POST",
                        "description": "Process uploaded file",
                        "parameters": {
                            "file_id": {"type": "string", "required": True, "description": "File ID"},
                            "options": {"type": "object", "required": False, "description": "Processing options"}
                        },
                        "responses": {
                            "200": {"description": "Processing completed"},
                            "404": {"description": "File not found"},
                            "500": {"description": "Processing error"}
                        }
                    }
                ]
            },
            "compression": {
                "base_path": "/api/v1/compression",
                "description": "Compression pipeline operations",
                "endpoints": [
                    {
                        "path": "/compress",
                        "method": "POST",
                        "description": "Compress data using holographic compression",
                        "parameters": {
                            "data": {"type": "string", "required": True, "description": "Data to compress"},
                            "algorithm": {"type": "string", "required": False, "description": "Compression algorithm"},
                            "threshold": {"type": "float", "required": False, "description": "Compression threshold"}
                        },
                        "responses": {
                            "200": {"description": "Data compressed successfully"},
                            "400": {"description": "Invalid data"},
                            "500": {"description": "Compression error"}
                        }
                    },
                    {
                        "path": "/decompress",
                        "method": "POST",
                        "description": "Decompress holographic data",
                        "parameters": {
                            "compressed_data": {"type": "string", "required": True, "description": "Compressed data"},
                            "algorithm": {"type": "string", "required": False, "description": "Decompression algorithm"}
                        },
                        "responses": {
                            "200": {"description": "Data decompressed successfully"},
                            "400": {"description": "Invalid compressed data"},
                            "500": {"description": "Decompression error"}
                        }
                    }
                ]
            },
            "monitoring": {
                "base_path": "/api/v1/monitoring",
                "description": "System monitoring and health checks",
                "endpoints": [
                    {
                        "path": "/health",
                        "method": "GET",
                        "description": "Get system health status",
                        "responses": {
                            "200": {"description": "Health status"},
                            "503": {"description": "Service unavailable"}
                        }
                    },
                    {
                        "path": "/metrics",
                        "method": "GET",
                        "description": "Get system metrics",
                        "parameters": {
                            "time_range": {"type": "string", "required": False, "description": "Time range for metrics"}
                        },
                        "responses": {
                            "200": {"description": "System metrics"},
                            "500": {"description": "Metrics unavailable"}
                        }
                    },
                    {
                        "path": "/alerts",
                        "method": "GET",
                        "description": "Get active alerts",
                        "responses": {
                            "200": {"description": "Active alerts"},
                            "500": {"description": "Alerts unavailable"}
                        }
                    }
                ]
            }
        },
        "authentication": {
            "type": "Bearer Token",
            "description": "API uses Bearer token authentication",
            "header": "Authorization: Bearer <token>"
        },
        "rate_limiting": {
            "requests_per_minute": 100,
            "burst_limit": 200,
            "description": "Rate limiting is applied per IP address"
        },
        "error_codes": {
            "400": "Bad Request - Invalid input parameters",
            "401": "Unauthorized - Invalid or missing authentication",
            "403": "Forbidden - Insufficient permissions",
            "404": "Not Found - Resource not found",
            "429": "Too Many Requests - Rate limit exceeded",
            "500": "Internal Server Error - Server error",
            "503": "Service Unavailable - Service temporarily unavailable"
        },
        "examples": {
            "store_data": {
                "request": {
                    "method": "POST",
                    "url": "/api/v1/holographic-memory/store",
                    "headers": {"Authorization": "Bearer <token>", "Content-Type": "application/json"},
                    "body": {"data": "Hello, Holographic Memory!", "metadata": {"type": "text"}}
                },
                "response": {
                    "status": 200,
                    "body": {"id": "mem_12345", "status": "stored", "timestamp": "2023-01-01T00:00:00Z"}
                }
            },
            "search_memory": {
                "request": {
                    "method": "POST",
                    "url": "/api/v1/holographic-memory/search",
                    "headers": {"Authorization": "Bearer <token>", "Content-Type": "application/json"},
                    "body": {"query": "holographic", "similarity_threshold": 0.8}
                },
                "response": {
                    "status": 200,
                    "body": {"results": [{"id": "mem_12345", "similarity": 0.95, "data": "Hello, Holographic Memory!"}]}
                }
            }
        }
    }
    
    return api_docs

def save_documentation(api_docs: Dict[str, Any]):
    """Save API documentation to files"""
    
    # Save as JSON
    with open("api_documentation.json", "w") as f:
        json.dump(api_docs, f, indent=2)
    
    # Generate OpenAPI/Swagger spec
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": api_docs["title"],
            "version": api_docs["version"],
            "description": api_docs["description"]
        },
        "servers": [{"url": api_docs["base_url"]}],
        "paths": {},
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
        }
    }
    
    # Convert endpoints to OpenAPI format
    for service_name, service in api_docs["endpoints"].items():
        for endpoint in service["endpoints"]:
            path = f"{service['base_path']}{endpoint['path']}"
            openapi_spec["paths"][path] = {
                endpoint["method"].lower(): {
                    "summary": endpoint["description"],
                    "parameters": [
                        {
                            "name": param_name,
                            "in": "query" if endpoint["method"] == "GET" else "body",
                            "required": param_info["required"],
                            "schema": {"type": param_info["type"]},
                            "description": param_info["description"]
                        }
                        for param_name, param_info in endpoint.get("parameters", {}).items()
                    ],
                    "responses": endpoint["responses"]
                }
            }
    
    with open("openapi_spec.json", "w") as f:
        json.dump(openapi_spec, f, indent=2)
    
    # Generate Markdown documentation
    markdown_doc = f"""# {api_docs['title']}

**Version:** {api_docs['version']}  
**Generated:** {api_docs['generated_at']}  
**Base URL:** {api_docs['base_url']}

## Overview

{api_docs['description']}

## Authentication

{api_docs['authentication']['description']}

**Header:** `{api_docs['authentication']['header']}`

## Rate Limiting

- **Requests per minute:** {api_docs['rate_limiting']['requests_per_minute']}
- **Burst limit:** {api_docs['rate_limiting']['burst_limit']}
- **Description:** {api_docs['rate_limiting']['description']}

## API Endpoints

"""
    
    for service_name, service in api_docs["endpoints"].items():
        markdown_doc += f"### {service_name.replace('_', ' ').title()}\n\n"
        markdown_doc += f"**Base Path:** `{service['base_path']}`\n\n"
        markdown_doc += f"{service['description']}\n\n"
        
        for endpoint in service["endpoints"]:
            markdown_doc += f"#### {endpoint['method']} {endpoint['path']}\n\n"
            markdown_doc += f"{endpoint['description']}\n\n"
            
            if endpoint.get("parameters"):
                markdown_doc += "**Parameters:**\n\n"
                for param_name, param_info in endpoint["parameters"].items():
                    required = "✓" if param_info["required"] else "○"
                    markdown_doc += f"- `{param_name}` ({param_info['type']}) {required} - {param_info['description']}\n"
                markdown_doc += "\n"
            
            markdown_doc += "**Responses:**\n\n"
            for status_code, response_info in endpoint["responses"].items():
                markdown_doc += f"- `{status_code}` - {response_info['description']}\n"
            markdown_doc += "\n"
    
    markdown_doc += """## Error Codes

"""
    for code, description in api_docs["error_codes"].items():
        markdown_doc += f"- `{code}` - {description}\n"
    
    markdown_doc += """
## Examples

### Store Data

```bash
curl -X POST {base_url}/api/v1/holographic-memory/store \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json" \\
  -d '{{"data": "Hello, Holographic Memory!", "metadata": {{"type": "text"}}}}'
```

### Search Memory

```bash
curl -X POST {base_url}/api/v1/holographic-memory/search \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "holographic", "similarity_threshold": 0.8}}'
```

## SDKs and Tools

- **OpenAPI Specification:** [openapi_spec.json](openapi_spec.json)
- **JSON Documentation:** [api_documentation.json](api_documentation.json)
- **Interactive Documentation:** Available at `/docs` endpoint when API is running

""".format(base_url=api_docs['base_url'])
    
    with open("API_DOCUMENTATION.md", "w") as f:
        f.write(markdown_doc)
    
    print("✅ API documentation generated:")
    print("   - api_documentation.json")
    print("   - openapi_spec.json") 
    print("   - API_DOCUMENTATION.md")

def main():
    """Main API documentation generation function"""
    print("🚀 Starting API Documentation Generation")
    print("=" * 60)
    
    # Generate documentation
    api_docs = generate_api_documentation()
    
    # Save documentation
    save_documentation(api_docs)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 API DOCUMENTATION REPORT")
    print("=" * 60)
    
    print(f"\n🔧 DOCUMENTATION GENERATED:")
    print(f"   Services Documented: {len(api_docs['endpoints'])}")
    print(f"   Total Endpoints: {sum(len(service['endpoints']) for service in api_docs['endpoints'].values())}")
    print(f"   Authentication: {api_docs['authentication']['type']}")
    print(f"   Rate Limiting: {api_docs['rate_limiting']['requests_per_minute']} req/min")
    
    print(f"\n📚 DOCUMENTATION FORMATS:")
    print(f"   - JSON API Documentation")
    print(f"   - OpenAPI 3.0 Specification")
    print(f"   - Markdown Documentation")
    
    print(f"\n🎯 SERVICES DOCUMENTED:")
    for service_name in api_docs['endpoints'].keys():
        print(f"   - {service_name.replace('_', ' ').title()}")
    
    print(f"\n🎉 API DOCUMENTATION COMPLETED SUCCESSFULLY!")
    print("✅ Comprehensive API documentation generated")
    print("✅ Multiple formats supported")
    print("✅ Authentication and rate limiting documented")
    print("✅ Examples and error codes included")
    print("✅ OpenAPI specification created")
    
    print("=" * 60)
    
    return api_docs

if __name__ == "__main__":
    main()
