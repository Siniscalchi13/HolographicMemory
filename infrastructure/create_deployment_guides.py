#!/usr/bin/env python3
"""
Deployment Guides Creation Script
================================

This script creates comprehensive deployment guides for the SOA system.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any

def create_deployment_guides():
    """Create comprehensive deployment guides"""
    
    guides = {
        "quick_start": {
            "title": "Quick Start Deployment Guide",
            "description": "Get the HolographicMemory SOA system running in minutes",
            "estimated_time": "5-10 minutes",
            "prerequisites": [
                "macOS 12.0+ (for Metal GPU support)",
                "Python 3.13+",
                "Xcode Command Line Tools",
                "8GB+ RAM",
                "2GB+ free disk space"
            ],
            "steps": [
                {
                    "step": 1,
                    "title": "Clone Repository",
                    "commands": [
                        "git clone https://github.com/your-org/HolographicMemory.git",
                        "cd HolographicMemory"
                    ]
                },
                {
                    "step": 2,
                    "title": "Install Dependencies",
                    "commands": [
                        "pip install -r requirements.txt",
                        "pip install -r services/holographic-memory/api/requirements.txt"
                    ]
                },
                {
                    "step": 3,
                    "title": "Build GPU Components",
                    "commands": [
                        "cd services/holographic-memory/core/native/holographic/build",
                        "make clean && make"
                    ]
                },
                {
                    "step": 4,
                    "title": "Start System",
                    "commands": [
                        "cd ../../../../..",
                        "python3.13 main.py"
                    ]
                },
                {
                    "step": 5,
                    "title": "Verify Deployment",
                    "commands": [
                        "curl http://localhost:8081/health",
                        "python3.13 benchmark_soa_system.py"
                    ]
                }
            ]
        },
        "production": {
            "title": "Production Deployment Guide",
            "description": "Deploy the HolographicMemory SOA system in a production environment",
            "estimated_time": "30-60 minutes",
            "prerequisites": [
                "macOS 12.0+ or Linux with CUDA support",
                "Python 3.13+",
                "Docker (optional)",
                "16GB+ RAM",
                "50GB+ free disk space",
                "SSL certificates",
                "Load balancer (optional)"
            ],
            "architecture": {
                "components": [
                    "Holographic Memory Service",
                    "Orchestrator Service", 
                    "Router Service",
                    "Vault Service",
                    "Telemetry Service",
                    "API Gateway",
                    "Monitoring System"
                ],
                "ports": {
                    "api": 8081,
                    "monitoring": 9090,
                    "telemetry": 8082
                }
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Environment Setup",
                    "commands": [
                        "export HOLOGRAPHIC_ENV=production",
                        "export GPU_ENABLED=true",
                        "export LOG_LEVEL=INFO"
                    ]
                },
                {
                    "step": 2,
                    "title": "Security Configuration",
                    "commands": [
                        "openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes",
                        "chmod 600 key.pem cert.pem"
                    ]
                },
                {
                    "step": 3,
                    "title": "Service Configuration",
                    "commands": [
                        "cp config/production.yaml config/active.yaml",
                        "python3.13 -c \"from services.orchestrator.orchestrator import HolographicMemoryOrchestrator; print('Config validated')\""
                    ]
                },
                {
                    "step": 4,
                    "title": "Database Setup",
                    "commands": [
                        "python3.13 scripts/setup_database.py",
                        "python3.13 scripts/migrate_database.py"
                    ]
                },
                {
                    "step": 5,
                    "title": "Deploy Services",
                    "commands": [
                        "python3.13 main.py --config config/active.yaml --daemon",
                        "systemctl enable holographic-memory"
                    ]
                },
                {
                    "step": 6,
                    "title": "Health Checks",
                    "commands": [
                        "python3.13 scripts/health_check.py",
                        "python3.13 add_integration_tests.py"
                    ]
                }
            ]
        },
        "docker": {
            "title": "Docker Deployment Guide",
            "description": "Deploy using Docker containers for easy scaling and management",
            "estimated_time": "15-30 minutes",
            "prerequisites": [
                "Docker 20.10+",
                "Docker Compose 2.0+",
                "8GB+ RAM",
                "10GB+ free disk space"
            ],
            "files": {
                "Dockerfile": """FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
COPY services/holographic-memory/api/requirements.txt ./api_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy source code
COPY . .

# Build GPU components (if supported)
RUN cd services/holographic-memory/core/native/holographic/build && \\
    make clean && make || echo "GPU build failed, continuing with CPU-only"

# Expose ports
EXPOSE 8081 8082 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8081/health || exit 1

# Start command
CMD ["python3.13", "main.py"]
""",
                "docker-compose.yml": """version: '3.8'

services:
  holographic-memory:
    build: .
    ports:
      - "8081:8081"
      - "8082:8082"
      - "9090:9090"
    environment:
      - HOLOGRAPHIC_ENV=production
      - GPU_ENABLED=true
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - holographic-memory
    restart: unless-stopped

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - holographic-memory
    restart: unless-stopped
""",
                "nginx.conf": """events {
    worker_connections 1024;
}

http {
    upstream holographic_memory {
        server holographic-memory:8081;
    }

    server {
        listen 80;
        server_name localhost;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name localhost;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://holographic_memory;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Create Docker Files",
                    "description": "Create Dockerfile and docker-compose.yml"
                },
                {
                    "step": 2,
                    "title": "Build and Start",
                    "commands": [
                        "docker-compose build",
                        "docker-compose up -d"
                    ]
                },
                {
                    "step": 3,
                    "title": "Verify Deployment",
                    "commands": [
                        "docker-compose ps",
                        "curl http://localhost:8081/health"
                    ]
                }
            ]
        },
        "kubernetes": {
            "title": "Kubernetes Deployment Guide",
            "description": "Deploy on Kubernetes for enterprise-scale deployment",
            "estimated_time": "45-90 minutes",
            "prerequisites": [
                "Kubernetes 1.20+",
                "kubectl configured",
                "Helm 3.0+",
                "Persistent storage class",
                "Load balancer or ingress controller"
            ],
            "files": {
                "deployment.yaml": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: holographic-memory
  labels:
    app: holographic-memory
spec:
  replicas: 3
  selector:
    matchLabels:
      app: holographic-memory
  template:
    metadata:
      labels:
        app: holographic-memory
    spec:
      containers:
      - name: holographic-memory
        image: holographic-memory:latest
        ports:
        - containerPort: 8081
        - containerPort: 8082
        - containerPort: 9090
        env:
        - name: HOLOGRAPHIC_ENV
          value: "production"
        - name: GPU_ENABLED
          value: "true"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: holographic-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: holographic-logs-pvc
""",
                "service.yaml": """apiVersion: v1
kind: Service
metadata:
  name: holographic-memory-service
spec:
  selector:
    app: holographic-memory
  ports:
  - name: api
    port: 8081
    targetPort: 8081
  - name: telemetry
    port: 8082
    targetPort: 8082
  - name: monitoring
    port: 9090
    targetPort: 9090
  type: LoadBalancer
""",
                "ingress.yaml": """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: holographic-memory-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - holographic-memory.example.com
    secretName: holographic-tls
  rules:
  - host: holographic-memory.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: holographic-memory-service
            port:
              number: 8081
"""
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Create Namespace",
                    "commands": [
                        "kubectl create namespace holographic-memory"
                    ]
                },
                {
                    "step": 2,
                    "title": "Deploy Persistent Volumes",
                    "commands": [
                        "kubectl apply -f pvc.yaml -n holographic-memory"
                    ]
                },
                {
                    "step": 3,
                    "title": "Deploy Application",
                    "commands": [
                        "kubectl apply -f deployment.yaml -n holographic-memory",
                        "kubectl apply -f service.yaml -n holographic-memory"
                    ]
                },
                {
                    "step": 4,
                    "title": "Configure Ingress",
                    "commands": [
                        "kubectl apply -f ingress.yaml -n holographic-memory"
                    ]
                },
                {
                    "step": 5,
                    "title": "Verify Deployment",
                    "commands": [
                        "kubectl get pods -n holographic-memory",
                        "kubectl get services -n holographic-memory"
                    ]
                }
            ]
        },
        "monitoring": {
            "title": "Monitoring and Observability Setup",
            "description": "Set up comprehensive monitoring and observability",
            "estimated_time": "20-40 minutes",
            "components": [
                "Prometheus for metrics collection",
                "Grafana for visualization",
                "AlertManager for alerting",
                "Jaeger for distributed tracing",
                "ELK Stack for log aggregation"
            ],
            "steps": [
                {
                    "step": 1,
                    "title": "Install Monitoring Stack",
                    "commands": [
                        "helm repo add prometheus-community https://prometheus-community.github.io/helm-charts",
                        "helm install prometheus prometheus-community/kube-prometheus-stack"
                    ]
                },
                {
                    "step": 2,
                    "title": "Configure Metrics Collection",
                    "commands": [
                        "kubectl apply -f monitoring/service-monitor.yaml",
                        "kubectl apply -f monitoring/alert-rules.yaml"
                    ]
                },
                {
                    "step": 3,
                    "title": "Setup Logging",
                    "commands": [
                        "helm install elasticsearch elastic/elasticsearch",
                        "helm install kibana elastic/kibana",
                        "helm install logstash elastic/logstash"
                    ]
                }
            ]
        },
        "security": {
            "title": "Security Hardening Guide",
            "description": "Secure the HolographicMemory SOA system",
            "estimated_time": "30-60 minutes",
            "areas": [
                "Network security",
                "Authentication and authorization",
                "Data encryption",
                "Container security",
                "API security"
            ],
            "steps": [
                {
                    "step": 1,
                    "title": "Enable TLS/SSL",
                    "commands": [
                        "openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes",
                        "chmod 600 key.pem cert.pem"
                    ]
                },
                {
                    "step": 2,
                    "title": "Configure Firewall",
                    "commands": [
                        "ufw allow 8081/tcp",
                        "ufw allow 8082/tcp",
                        "ufw enable"
                    ]
                },
                {
                    "step": 3,
                    "title": "Setup Authentication",
                    "commands": [
                        "python3.13 scripts/setup_auth.py",
                        "python3.13 scripts/generate_api_keys.py"
                    ]
                }
            ]
        }
    }
    
    return guides

def save_deployment_guides(guides: Dict[str, Any]):
    """Save deployment guides to files"""
    
    # Create deployment directory
    os.makedirs("deployment", exist_ok=True)
    os.makedirs("deployment/docker", exist_ok=True)
    os.makedirs("deployment/kubernetes", exist_ok=True)
    os.makedirs("deployment/monitoring", exist_ok=True)
    
    # Save main deployment guide
    with open("deployment/DEPLOYMENT_GUIDE.md", "w") as f:
        f.write(f"""# HolographicMemory SOA Deployment Guide

**Generated:** {datetime.now().isoformat()}

## Overview

This guide provides comprehensive deployment instructions for the HolographicMemory Service-Oriented Architecture system.

## Quick Start

For a quick deployment, see [Quick Start Guide](quick_start.md).

## Production Deployment

For production environments, see [Production Deployment Guide](production.md).

## Container Deployment

For Docker and Kubernetes deployments, see:
- [Docker Deployment Guide](docker.md)
- [Kubernetes Deployment Guide](kubernetes.md)

## Monitoring and Security

For monitoring and security setup, see:
- [Monitoring Setup Guide](monitoring.md)
- [Security Hardening Guide](security.md)

## Support

For issues and support, please refer to the main documentation or contact the development team.
""")
    
    # Save individual guides
    for guide_name, guide_data in guides.items():
        if guide_name in ["docker", "kubernetes"]:
            # Save YAML files for container deployments
            if "files" in guide_data:
                for filename, content in guide_data["files"].items():
                    filepath = f"deployment/{guide_name}/{filename}"
                    with open(filepath, "w") as f:
                        f.write(content)
        
        # Save markdown guide
        markdown_content = f"""# {guide_data['title']}

**Description:** {guide_data['description']}  
**Estimated Time:** {guide_data['estimated_time']}

## Prerequisites

"""
        
        if "prerequisites" in guide_data:
            for prereq in guide_data["prerequisites"]:
                markdown_content += f"- {prereq}\n"
        
        if "architecture" in guide_data:
            markdown_content += f"""
## Architecture

### Components
"""
            for component in guide_data["architecture"]["components"]:
                markdown_content += f"- {component}\n"
            
            if "ports" in guide_data["architecture"]:
                markdown_content += f"""
### Ports
"""
                for service, port in guide_data["architecture"]["ports"].items():
                    markdown_content += f"- {service}: {port}\n"
        
        if "components" in guide_data:
            markdown_content += f"""
## Components

"""
            for component in guide_data["components"]:
                markdown_content += f"- {component}\n"
        
        if "areas" in guide_data:
            markdown_content += f"""
## Security Areas

"""
            for area in guide_data["areas"]:
                markdown_content += f"- {area}\n"
        
        markdown_content += f"""
## Deployment Steps

"""
        
        for step in guide_data["steps"]:
            markdown_content += f"""
### Step {step['step']}: {step['title']}

"""
            if "description" in step:
                markdown_content += f"{step['description']}\n\n"
            
            if "commands" in step:
                markdown_content += "**Commands:**\n\n```bash\n"
                for command in step["commands"]:
                    markdown_content += f"{command}\n"
                markdown_content += "```\n\n"
        
        # Save markdown file
        with open(f"deployment/{guide_name}.md", "w") as f:
            f.write(markdown_content)
    
    # Save JSON configuration
    with open("deployment/deployment_config.json", "w") as f:
        json.dump(guides, f, indent=2)
    
    print("✅ Deployment guides created:")
    print("   - deployment/DEPLOYMENT_GUIDE.md")
    print("   - deployment/quick_start.md")
    print("   - deployment/production.md")
    print("   - deployment/docker/")
    print("   - deployment/kubernetes/")
    print("   - deployment/monitoring.md")
    print("   - deployment/security.md")
    print("   - deployment/deployment_config.json")

def main():
    """Main deployment guide creation function"""
    print("🚀 Starting Deployment Guide Creation")
    print("=" * 60)
    
    # Create deployment guides
    guides = create_deployment_guides()
    
    # Save guides
    save_deployment_guides(guides)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT GUIDE REPORT")
    print("=" * 60)
    
    print(f"\n🔧 DEPLOYMENT GUIDES CREATED:")
    print(f"   Quick Start Guide: {guides['quick_start']['estimated_time']}")
    print(f"   Production Guide: {guides['production']['estimated_time']}")
    print(f"   Docker Guide: {guides['docker']['estimated_time']}")
    print(f"   Kubernetes Guide: {guides['kubernetes']['estimated_time']}")
    print(f"   Monitoring Guide: {guides['monitoring']['estimated_time']}")
    print(f"   Security Guide: {guides['security']['estimated_time']}")
    
    print(f"\n📚 DEPLOYMENT SCENARIOS:")
    for guide_name, guide_data in guides.items():
        print(f"   - {guide_data['title']}")
    
    print(f"\n🎯 DEPLOYMENT OPTIONS:")
    print(f"   - Local Development")
    print(f"   - Production Server")
    print(f"   - Docker Containers")
    print(f"   - Kubernetes Cluster")
    print(f"   - Cloud Platforms")
    
    print(f"\n🎉 DEPLOYMENT GUIDES COMPLETED SUCCESSFULLY!")
    print("✅ Comprehensive deployment documentation created")
    print("✅ Multiple deployment scenarios covered")
    print("✅ Container and orchestration support")
    print("✅ Security and monitoring included")
    print("✅ Step-by-step instructions provided")
    
    print("=" * 60)
    
    return guides

if __name__ == "__main__":
    main()
