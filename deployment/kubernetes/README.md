# Kubernetes Configuration Directory

## Overview

The `/deployment/kubernetes/` directory contains Kubernetes manifests for enterprise-scale orchestration of the HolographicMemory SOA system. This directory provides production-ready Kubernetes configurations with high availability, auto-scaling, and enterprise networking capabilities.

## Directory Structure

```
kubernetes/
├── README.md              # This comprehensive guide
├── deployment.yaml        # Kubernetes deployment manifest
├── service.yaml           # Kubernetes service manifest
└── ingress.yaml           # Kubernetes ingress manifest
```

## File Details

### `deployment.yaml`
**Purpose**: Kubernetes deployment manifest for application orchestration
**Technical Details**:

#### API Version and Kind
- **API Version**: `apps/v1` (stable Kubernetes API)
- **Kind**: `Deployment` (manages replica sets and pods)

#### Metadata
- **Name**: `holographic-memory`
- **Labels**: `app: holographic-memory` (for service discovery)

#### Deployment Specification
- **Replicas**: `3` (high availability with 3 instances)
- **Selector**: `matchLabels: app: holographic-memory`
- **Strategy**: Rolling update (default, zero-downtime deployments)

#### Pod Template Specification
- **Labels**: `app: holographic-memory` (consistent labeling)

#### Container Configuration
- **Name**: `holographic-memory`
- **Image**: `holographic-memory:latest` (container image)
- **Ports**:
  - `8081` - Main API service
  - `8082` - Telemetry and metrics
  - `9090` - Monitoring and observability

#### Environment Variables
- **HOLOGRAPHIC_ENV**: `"production"` (production environment)
- **GPU_ENABLED**: `"true"` (GPU acceleration enabled)

#### Resource Management
- **Requests** (guaranteed resources):
  - **Memory**: `4Gi` (4 gigabytes)
  - **CPU**: `2` (2 CPU cores)
- **Limits** (maximum resources):
  - **Memory**: `8Gi` (8 gigabytes)
  - **CPU**: `4` (4 CPU cores)

#### Health Checks
- **Liveness Probe**:
  - **Type**: HTTP GET
  - **Path**: `/health`
  - **Port**: `8081`
  - **Initial Delay**: `30 seconds`
  - **Period**: `10 seconds`
  - **Purpose**: Restart container if unhealthy
- **Readiness Probe**:
  - **Type**: HTTP GET
  - **Path**: `/health`
  - **Port**: `8081`
  - **Initial Delay**: `5 seconds`
  - **Period**: `5 seconds`
  - **Purpose**: Remove from service if not ready

#### Volume Mounts
- **Data Volume**:
  - **Name**: `data-volume`
  - **Mount Path**: `/app/data`
  - **Purpose**: Persistent data storage
- **Logs Volume**:
  - **Name**: `logs-volume`
  - **Mount Path**: `/app/logs`
  - **Purpose**: Log file persistence

#### Volumes
- **Data Volume**:
  - **Type**: `persistentVolumeClaim`
  - **Claim Name**: `holographic-data-pvc`
- **Logs Volume**:
  - **Type**: `persistentVolumeClaim`
  - **Claim Name**: `holographic-logs-pvc`

### `service.yaml`
**Purpose**: Kubernetes service manifest for network access
**Technical Details**:

#### API Version and Kind
- **API Version**: `v1` (core Kubernetes API)
- **Kind**: `Service` (network access to pods)

#### Metadata
- **Name**: `holographic-memory-service`

#### Service Specification
- **Selector**: `app: holographic-memory` (matches deployment labels)
- **Type**: `LoadBalancer` (external access with load balancing)

#### Port Configuration
- **API Port**:
  - **Name**: `api`
  - **Port**: `8081`
  - **Target Port**: `8081`
  - **Protocol**: TCP (default)
- **Telemetry Port**:
  - **Name**: `telemetry`
  - **Port**: `8082`
  - **Target Port**: `8082`
  - **Protocol**: TCP
- **Monitoring Port**:
  - **Name**: `monitoring`
  - **Port**: `9090`
  - **Target Port**: `9090`
  - **Protocol**: TCP

#### Network Configuration
- **External Access**: Via LoadBalancer (cloud provider integration)
- **Internal Access**: ClusterIP (internal cluster communication)
- **Service Discovery**: DNS-based (`holographic-memory-service.namespace.svc.cluster.local`)

### `ingress.yaml`
**Purpose**: Kubernetes ingress manifest for external access and SSL termination
**Technical Details**:

#### API Version and Kind
- **API Version**: `networking.k8s.io/v1` (Ingress API v1)
- **Kind**: `Ingress` (HTTP/HTTPS routing)

#### Metadata
- **Name**: `holographic-memory-ingress`
- **Annotations**:
  - `nginx.ingress.kubernetes.io/ssl-redirect: "true"` (force HTTPS)
  - `nginx.ingress.kubernetes.io/force-ssl-redirect: "true"` (redirect HTTP to HTTPS)

#### TLS Configuration
- **Hosts**: `holographic-memory.example.com`
- **Secret Name**: `holographic-tls` (TLS certificate secret)
- **Purpose**: SSL/TLS termination at ingress level

#### Ingress Rules
- **Host**: `holographic-memory.example.com`
- **HTTP Paths**:
  - **Path**: `/` (root path)
  - **Path Type**: `Prefix` (matches all subpaths)
  - **Backend**:
    - **Service Name**: `holographic-memory-service`
    - **Service Port**: `8081` (API port)

#### Security Features
- **SSL/TLS Termination**: HTTPS encryption
- **Force HTTPS**: Automatic HTTP to HTTPS redirect
- **Host-based Routing**: Domain-based access control

## Deployment Instructions

### Prerequisites
- Kubernetes 1.20+
- kubectl configured and authenticated
- Helm 3.0+ (optional, for advanced deployments)
- Persistent storage class available
- Load balancer or ingress controller installed
- DNS configuration for ingress hostname

### Namespace Setup
```bash
# Create dedicated namespace
kubectl create namespace holographic-memory

# Set default namespace for kubectl
kubectl config set-context --current --namespace=holographic-memory
```

### Persistent Volume Claims
```yaml
# Create pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: holographic-data-pvc
  namespace: holographic-memory
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: holographic-logs-pvc
  namespace: holographic-memory
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

### TLS Certificate Setup
```bash
# Generate TLS certificate
openssl req -x509 -newkey rsa:4096 -keyout tls.key -out tls.crt -days 365 -nodes \
  -subj "/CN=holographic-memory.example.com"

# Create Kubernetes TLS secret
kubectl create secret tls holographic-tls \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=holographic-memory
```

### Deployment Steps
```bash
# 1. Deploy Persistent Volume Claims
kubectl apply -f pvc.yaml

# 2. Deploy the application
kubectl apply -f deployment.yaml

# 3. Deploy the service
kubectl apply -f service.yaml

# 4. Deploy the ingress
kubectl apply -f ingress.yaml

# 5. Verify deployment
kubectl get pods -n holographic-memory
kubectl get services -n holographic-memory
kubectl get ingress -n holographic-memory
```

### Verification Commands
```bash
# Check pod status
kubectl get pods -n holographic-memory

# Check service endpoints
kubectl get endpoints -n holographic-memory

# Check ingress status
kubectl describe ingress holographic-memory-ingress -n holographic-memory

# Test connectivity
kubectl port-forward svc/holographic-memory-service 8081:8081 -n holographic-memory
curl http://localhost:8081/health
```

## Configuration Options

### Resource Scaling
```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: holographic-memory-hpa
  namespace: holographic-memory
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: holographic-memory
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Network Policies
```yaml
# Network policy for security
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: holographic-memory-netpol
  namespace: holographic-memory
spec:
  podSelector:
    matchLabels:
      app: holographic-memory
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8081
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### ConfigMaps and Secrets
```yaml
# ConfigMap for application configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: holographic-memory-config
  namespace: holographic-memory
data:
  HOLOGRAPHIC_ENV: "production"
  GPU_ENABLED: "true"
  LOG_LEVEL: "INFO"
  GRID_SIZE: "64"

---
# Secret for sensitive data
apiVersion: v1
kind: Secret
metadata:
  name: holographic-memory-secrets
  namespace: holographic-memory
type: Opaque
data:
  api-key: <base64-encoded-api-key>
  db-password: <base64-encoded-password>
```

## Monitoring and Observability

### Service Monitor (Prometheus)
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: holographic-memory-monitor
  namespace: holographic-memory
spec:
  selector:
    matchLabels:
      app: holographic-memory
  endpoints:
  - port: monitoring
    path: /metrics
    interval: 30s
```

### Pod Disruption Budget
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: holographic-memory-pdb
  namespace: holographic-memory
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: holographic-memory
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n holographic-memory

# Describe pod for details
kubectl describe pod <pod-name> -n holographic-memory

# Check pod logs
kubectl logs <pod-name> -n holographic-memory

# Check events
kubectl get events -n holographic-memory --sort-by='.lastTimestamp'
```

#### Service Connectivity Issues
```bash
# Check service endpoints
kubectl get endpoints -n holographic-memory

# Test service connectivity
kubectl run test-pod --image=busybox -it --rm -- nslookup holographic-memory-service

# Port forward for testing
kubectl port-forward svc/holographic-memory-service 8081:8081 -n holographic-memory
```

#### Ingress Issues
```bash
# Check ingress status
kubectl describe ingress holographic-memory-ingress -n holographic-memory

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Test DNS resolution
nslookup holographic-memory.example.com
```

#### Persistent Volume Issues
```bash
# Check PVC status
kubectl get pvc -n holographic-memory

# Describe PVC for details
kubectl describe pvc holographic-data-pvc -n holographic-memory

# Check PV status
kubectl get pv
```

### Performance Optimization

#### Resource Tuning
```yaml
# Adjust resource requests/limits based on monitoring
resources:
  requests:
    memory: "2Gi"    # Reduce if over-provisioned
    cpu: "1"         # Adjust based on CPU usage
  limits:
    memory: "4Gi"    # Increase if memory constrained
    cpu: "2"         # Adjust based on CPU usage
```

#### Node Affinity
```yaml
# Schedule pods on specific nodes
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values:
          - gpu-enabled
```

## Security Considerations

### RBAC (Role-Based Access Control)
```yaml
# Service account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: holographic-memory-sa
  namespace: holographic-memory

---
# Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: holographic-memory-role
  namespace: holographic-memory
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]

---
# Role binding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: holographic-memory-rolebinding
  namespace: holographic-memory
subjects:
- kind: ServiceAccount
  name: holographic-memory-sa
  namespace: holographic-memory
roleRef:
  kind: Role
  name: holographic-memory-role
  apiGroup: rbac.authorization.k8s.io
```

### Pod Security Standards
```yaml
# Pod security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000
  seccompProfile:
    type: RuntimeDefault
```

## Maintenance

### Rolling Updates
```bash
# Update image
kubectl set image deployment/holographic-memory \
  holographic-memory=holographic-memory:v2.0.0 \
  -n holographic-memory

# Check rollout status
kubectl rollout status deployment/holographic-memory -n holographic-memory

# Rollback if needed
kubectl rollout undo deployment/holographic-memory -n holographic-memory
```

### Backup and Recovery
```bash
# Backup PVC data
kubectl exec -n holographic-memory <pod-name> -- tar czf /tmp/backup.tar.gz /app/data

# Copy backup from pod
kubectl cp holographic-memory/<pod-name>:/tmp/backup.tar.gz ./backup.tar.gz

# Restore data
kubectl cp ./backup.tar.gz holographic-memory/<pod-name>:/tmp/backup.tar.gz
kubectl exec -n holographic-memory <pod-name> -- tar xzf /tmp/backup.tar.gz -C /
```

## Support

### Documentation
- **Kubernetes Documentation**: [Kubernetes Official Docs](https://kubernetes.io/docs/)
- **Ingress Controllers**: [NGINX Ingress](https://kubernetes.github.io/ingress-nginx/)
- **Persistent Volumes**: [Storage Classes](https://kubernetes.io/docs/concepts/storage/storage-classes/)

### Monitoring Tools
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis

## Conclusion

The Kubernetes configuration directory provides enterprise-grade orchestration for the HolographicMemory SOA system. With high availability, auto-scaling, persistent storage, SSL termination, and comprehensive monitoring, this setup enables reliable, scalable, and secure deployment of the holographic memory system in any Kubernetes environment.

The configuration supports both cloud and on-premises deployments, with advanced features like horizontal pod autoscaling, network policies, and pod disruption budgets for enterprise-grade reliability and security.
