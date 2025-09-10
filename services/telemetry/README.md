# Telemetry Service

## Purpose
Track compression metrics and per-layer loads to inform math_core optimizations. Provides comprehensive real-time monitoring for the entire SOA system.

## Core APIs (internal)
- `PerformanceTelemetry.track_compression(original: int, stored: int, layer: str) -> None`
- `PerformanceTelemetry.suggest_rebalancing(importance: dict[str,float], total_budget: int, floors: dict[str,int] | None = None) -> dict[str,int]`

## Real-time Monitoring
- `RealTimeMonitor` - Comprehensive system monitoring with multi-threaded metric collection
- `WebSocketServer` - Real-time data streaming for live dashboard updates

### Monitored Components
- **System Metrics**: CPU usage, memory usage, disk usage, network I/O
- **GPU Metrics**: GPU availability, usage percentage, memory usage, shader status
- **Service Metrics**: Health status, uptime, request counts for all SOA services
- **Compression Metrics**: Algorithm performance, compression ratios, usage statistics
- **Performance Metrics**: Response times, throughput, error rates, queue lengths

### Alert System
- Configurable thresholds for critical metrics
- Multi-severity alerting (warning, error, critical)
- Alert resolution tracking
- Historical alert retention

### WebSocket Streaming
- Real-time metric updates to connected clients
- Historical data queries
- Alert management via WebSocket API

## Dependencies
- `prometheus_client>=0.19.0` - Metrics collection
- `psutil>=5.9.0` - System metrics
- `websockets>=15.0.0` - Real-time streaming
- `numpy>=1.26.0` - Numerical operations

## Usage
```python
from services.telemetry.realtime_monitor import RealTimeMonitor

# Start monitoring
monitor = RealTimeMonitor()
monitor.start_monitoring()

# Get current metrics
metrics = monitor.get_current_metrics()

# Get alerts
alerts = monitor.get_alerts()

# Stop monitoring
monitor.stop_monitoring()
```

