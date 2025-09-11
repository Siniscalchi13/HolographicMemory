from __future__ import annotations

import psutil
import time
from typing import Dict, Any


def collect_metrics() -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    net = psutil.net_io_counters()
    try:
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
    except Exception:
        disk_percent = 0.0
    try:
        conns = len(psutil.net_connections())
    except Exception:
        conns = 0

    # GPU placeholder (extend with actual GPU provider integration)
    gpu_util = 0.0
    gpu_available = False

    return {
        "ts": time.time(),
        "cpu": cpu,
        "memory": {"used": vm.used, "total": vm.total, "percent": vm.percent},
        "network": {"bytes_sent": net.bytes_sent, "bytes_recv": net.bytes_recv},
        "disk": {"percent": disk_percent},
        "connections": conns,
        "gpu": {"utilization": gpu_util, "available": gpu_available},
        # Back-compat alternate names some UIs might expect
        "cpu_percent": cpu,
        "memory_percent": vm.percent,
        "disk_percent": disk_percent,
        "net_sent": net.bytes_sent,
        "net_recv": net.bytes_recv,
    }
