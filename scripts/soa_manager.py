#!/usr/bin/env python3
"""
SOA Process Manager
==================

Intelligent process and port management for the HolographicMemory SOA system.
- Automatically detects and kills existing processes
- Dynamically finds available ports
- Manages service lifecycle
- Provides clean startup/shutdown
"""

import sys
import time
import socket
import subprocess
import psutil
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    name: str
    command: str
    port: Optional[int] = None
    health_check_path: Optional[str] = None
    startup_delay: float = 2.0
    required: bool = True

class SOAManager:
    """Manages the entire SOA system lifecycle"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.venv_python = self.project_root / "venv313" / "bin" / "python"
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.service_ports: Dict[str, int] = {}
        self.config_file = self.project_root / "soa_config.json"
        self.service_started_at: Dict[str, float] = {}
        
        # Service configurations
        self.services = {
            "main_dashboard": ServiceConfig(
                name="SOA Dashboard Hub",
                command=f"cd {self.project_root} && {self.venv_python} soa_config_server.py {{port}}",
                health_check_path="/soa_dashboard.html",
                startup_delay=2.0
            ),
            "api": ServiceConfig(
                name="API Server",
                command=f"{self.venv_python} main.py --port {{port}}",
                health_check_path="/healthz",
                startup_delay=3.0
            ),
            "terminal_service": ServiceConfig(
                name="Terminal Service",
                command=f"cd {self.project_root} && {self.venv_python} -m uvicorn services.terminal.terminal_server:app --port {{port}}",
                health_check_path="/healthz",
                startup_delay=2.0
            ),
            "terminal_interface": ServiceConfig(
                name="Terminal Interface", 
                command=f"cd {self.project_root}/services/dashboard/web && python3 -m http.server {{port}}",
                health_check_path="/",
                startup_delay=1.0
            ),
            "metrics_service": ServiceConfig(
                name="Metrics Service",
                command=f"cd {self.project_root} && {self.venv_python} -m uvicorn services.metrics.server:app --port {{port}}",
                health_check_path="/healthz",
                startup_delay=2.0
            ),
            "analytics_dashboard": ServiceConfig(
                name="Analytics Dashboard",
                command=f"cd {self.project_root}/dashboard && {self.venv_python} app.py --port {{port}}",
                health_check_path="/",
                startup_delay=2.0,
                required=True
            ),
            "api_status": ServiceConfig(
                name="API Status Page",
                command=f"cd {self.project_root}/services/holographic-memory/api/static && python3 -m http.server {{port}}",
                health_check_path="/",
                startup_delay=1.0,
                required=False
            )
        }
        
    def find_free_port(self, start_port: int = 8080, max_attempts: int = 50) -> int:
        """Find the next available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_attempts}")
    
    def kill_existing_processes(self) -> None:
        """Kill any existing HolographicMemory processes"""
        logger.info("🔍 Checking for existing processes...")
        
        killed_processes = []
        
        # Find processes by command patterns and working directory
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                cwd = proc.info.get('cwd', '')
                
                # Check if it's a HolographicMemory process
                is_holo_process = (
                    'main.py' in cmdline or
                    'app_soa' in cmdline or
                    'http.server' in cmdline or
                    'app.py' in cmdline
                ) and (
                    'HolographicMemory' in cwd or
                    'HolographicMemory' in cmdline or
                    'holographic' in cmdline.lower()
                )
                
                if is_holo_process:
                    logger.info("🛑 Killing process %s: %s...", proc.info['pid'], cmdline[:100])
                    proc.kill()
                    killed_processes.append(proc.info['pid'])
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        if killed_processes:
            logger.info("✅ Killed %d existing processes", len(killed_processes))
            time.sleep(2)  # Give processes time to fully terminate
        else:
            logger.info("✅ No existing processes found")
    
    def assign_ports(self) -> None:
        """Dynamically assign ports to services"""
        logger.info("🔌 Assigning ports to services...")
        
        start_port = 8080
        for service_name, service_config in self.services.items():
            if service_config.port is None:
                port = self.find_free_port(start_port)
                self.service_ports[service_name] = port
                logger.info("   %s: port %d", service_name, port)
                start_port = port + 1
            else:
                self.service_ports[service_name] = service_config.port
                logger.info("   %s: port %d (fixed)", service_name, service_config.port)
    
    def start_service(self, service_name: str, service_config: ServiceConfig) -> bool:
        """Start a single service"""
        port = self.service_ports[service_name]
        command = service_config.command.format(port=port)
        
        logger.info("🚀 Starting %s on port %d...", service_config.name, port)
        
        try:
            # Start the process
            if service_name == "dashboard":
                # Special handling for dashboard (needs to run in specific directory)
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=self.project_root / "services" / "dashboard" / "web"
                )
            elif service_name == "api_static":
                # Special handling for API static (needs to run in specific directory)
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=self.project_root / "services" / "holographic-memory" / "api" / "static"
                )
            else:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=self.project_root
                )
            
            self.running_processes[service_name] = proc
            
            # Wait for startup
            time.sleep(service_config.startup_delay)
            
            # Health check
            if service_config.health_check_path:
                if self.health_check(port, service_config.health_check_path):
                    logger.info(f"✅ {service_config.name} started successfully")
                    # record start time
                    import time as _t
                    self.service_started_at[service_name] = _t.time()
                    return True
                else:
                    logger.error(f"❌ {service_config.name} failed health check")
                    return False
            else:
                logger.info(f"✅ {service_config.name} started (no health check)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to start {service_config.name}: {e}")
            return False
    
    def health_check(self, port: int, path: str, max_retries: int = 3) -> bool:
        """Perform health check on a service with retry logic"""
        try:
            import requests
            for attempt in range(max_retries):
                try:
                    response = requests.get(f"http://localhost:{port}{path}", timeout=5)
                    if response.status_code == 200:
                        return True
                    logger.warning(f"Health check attempt {attempt + 1}: HTTP {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Health check attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
            
            return False
        except ImportError:
            logger.warning("requests library not available, skipping health check")
            return True  # Assume healthy if we can't check
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all configured services"""
        logger.info("🌐 Starting SOA system...")
        
        # Kill existing processes
        self.kill_existing_processes()
        
        # Assign ports
        self.assign_ports()
        
        # Start services in order
        failed_services = []
        
        for service_name, service_config in self.services.items():
            if not self.start_service(service_name, service_config):
                if service_config.required:
                    failed_services.append(service_name)
                else:
                    logger.warning(f"⚠️  Optional service {service_name} failed to start")
        
        if failed_services:
            logger.error(f"❌ Failed to start required services: {failed_services}")
            return False
        
        # Save configuration
        self.save_config()
        
        logger.info("🎉 SOA system started successfully!")
        self.print_status()
        
        # Auto-open main dashboard
        self.open_main_dashboard()
        
        return True
    
    def stop_all_services(self) -> None:
        """Stop all running services"""
        logger.info("🛑 Stopping SOA system...")
        
        for service_name, proc in self.running_processes.items():
            try:
                logger.info(f"   Stopping {service_name}...")
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"   Force killing {service_name}...")
                proc.kill()
            except Exception as e:
                logger.error(f"   Error stopping {service_name}: {e}")
        
        self.running_processes.clear()
        self.service_ports.clear()
        
        # Also kill any remaining processes
        self.kill_existing_processes()
        
        logger.info("✅ SOA system stopped")
    
    def open_main_dashboard(self) -> None:
        """Automatically open the main dashboard in the default browser"""
        main_dashboard_port = self.service_ports.get("main_dashboard")
        if main_dashboard_port:
            dashboard_url = f"http://localhost:{main_dashboard_port}/soa_dashboard.html"
            logger.info("🌐 Opening main dashboard in browser...")
            
            try:
                import webbrowser
                webbrowser.open(dashboard_url)
                logger.info("✅ Dashboard opened successfully")
            except Exception as e:
                logger.warning("⚠️  Could not auto-open browser: %s", e)
                logger.info("💡 Please manually open: %s", dashboard_url)
        else:
            logger.warning("⚠️  Main dashboard port not found")
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        config = {
            "service_ports": self.service_ports,
            "timestamp": time.time(),
            "services": {name: {
                "name": conf.name,
                "port": self.service_ports.get(name),
                "running": name in self.running_processes,
                "pid": (self.running_processes.get(name).pid if name in self.running_processes else None),
                "started_at": self.service_started_at.get(name)
            } for name, conf in self.services.items()}
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Also save the enhanced port registry
        self.save_port_registry()
    
    def save_port_registry(self) -> None:
        """Save comprehensive port registry with URLs for navigation"""
        import datetime
        
        registry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "registry_version": "1.0",
            "services": {}
        }
        
        # Build comprehensive service registry
        for service_name, service_config in self.services.items():
            port = self.service_ports.get(service_name)
            if port:
                service_info = {
                    "name": service_config.name,
                    "port": port,
                    "running": service_name in self.running_processes,
                    "started_at": self.service_started_at.get(service_name)
                }
                
                # Add service-specific URLs
                if service_name == "main_dashboard":
                    service_info["url"] = f"http://localhost:{port}/soa_dashboard.html"
                    service_info["api_url"] = f"http://localhost:{port}/api/soa-config"
                elif service_name == "api":
                    service_info["url"] = f"http://localhost:{port}"
                    service_info["docs_url"] = f"http://localhost:{port}/docs"
                    service_info["health_url"] = f"http://localhost:{port}/healthz"
                elif service_name == "terminal_interface":
                    service_info["url"] = f"http://localhost:{port}"
                elif service_name == "analytics_dashboard":
                    service_info["url"] = f"http://localhost:{port}"
                elif service_name == "api_status":
                    service_info["url"] = f"http://localhost:{port}"
                elif service_name == "terminal_service":
                    service_info["ws_url"] = f"ws://localhost:{port}/ws"
                    service_info["health_url"] = f"http://localhost:{port}/healthz"
                elif service_name == "metrics_service":
                    service_info["ws_url"] = f"ws://localhost:{port}/ws"
                    service_info["health_url"] = f"http://localhost:{port}/healthz"
                
                registry["services"][service_name] = service_info
        
        # Save to both soa_ports.json and the config directory
        registry_file = self.project_root / "soa_ports.json"
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"📋 Port registry saved to {registry_file}")
    
    def load_config(self) -> Optional[Dict]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None
    
    def print_status(self) -> None:
        """Print current system status"""
        print("\n" + "="*60)
        print("🌌 🚀 HOLOGRAPHIC MEMORY SOA SYSTEM STATUS 🚀 🌌")
        print("="*60)
        
        for service_name, service_config in self.services.items():
            port = self.service_ports.get(service_name, "N/A")
            status = "🟢 Running" if service_name in self.running_processes else "🔴 Stopped"
            print(f"{service_config.name:<20} | Port {port:<6} | {status}")
        
        print("\n🌐 Access URLs:")
        for service_name, port in self.service_ports.items():
            if service_name == "main_dashboard":
                print(f"   🎯 MAIN DASHBOARD: http://localhost:{port}/soa_dashboard.html")
            elif service_name == "api":
                print(f"   API Server:     http://localhost:{port}")
                print(f"   API Docs:       http://localhost:{port}/docs")
            elif service_name == "dashboard":
                print(f"   Web Dashboard:  http://localhost:{port}")
            elif service_name == "enhanced_dashboard":
                print(f"   Enhanced UI:    http://localhost:{port}")
            elif service_name == "api_static":
                print(f"   API Static:     http://localhost:{port}")
        
        print("\n💡 🎮 COMMANDS:")
        print("   make soa-start    - 🚀 Start the system")
        print("   make soa-stop     - 🛑 Stop the system") 
        print("   make soa-status   - 📊 Show status")
        print("   make soa-restart  - 🔄 Restart the system")
        print("   ./soa start       - 🎯 Simple start")
        print("   ./soa clean       - 🧹 Fresh start")
        print("="*60)

    # --- Targeted service controls (single service) ---
    def _load_persisted(self) -> Dict:
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def start_single(self, service_name: str) -> bool:
        persisted = self._load_persisted()
        ports = persisted.get("service_ports", {})
        # Keep existing port if present
        if service_name in ports:
            self.service_ports[service_name] = ports[service_name]
        else:
            # assign new free port after existing
            start_port = max([*ports.values(), 8080]) + 1 if ports else 8080
            self.service_ports.update(ports)
            self.service_ports[service_name] = self.find_free_port(start_port)
        # Start only this service
        if service_name not in self.services:
            logger.error("Unknown service: %s", service_name)
            return False
        ok = self.start_service(service_name, self.services[service_name])
        self.save_config()
        return ok

    def stop_single(self, service_name: str) -> bool:
        persisted = self._load_persisted()
        svc_info = persisted.get("services", {}).get(service_name)
        if not svc_info:
            logger.error("No persisted info for service: %s", service_name)
            return False
        pid = svc_info.get("pid")
        if not pid:
            logger.error("No PID recorded for service: %s", service_name)
            return False
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=5)
            except psutil.TimeoutExpired:
                p.kill()
            # Refresh state
            persisted["services"][service_name]["running"] = False
            persisted["services"][service_name]["pid"] = None
            persisted["services"].setdefault(service_name, {})
            persisted["services"][service_name]["started_at"] = None
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(persisted, f, indent=2)
            return True
        except psutil.NoSuchProcess:
            logger.warning("Process already gone for %s", service_name)
            return True
        except Exception as e:
            logger.error("Failed to stop %s: %s", service_name, e)
            return False

    def restart_single(self, service_name: str) -> bool:
        self.stop_single(service_name)
        return self.start_single(service_name)
    
    def status(self) -> None:
        """Show current system status"""
        config = self.load_config()
        if config:
            print("📊 SOA System Status:")
            for service_name, service_info in config.get("services", {}).items():
                status = "🟢 Running" if service_info.get("running") else "🔴 Stopped"
                port = service_info.get("port", "N/A")
                port_str = str(port) if port is not None else "N/A"
                print(f"   {service_info['name']:<20} | Port {port_str:<6} | {status}")
        else:
            print("🔴 SOA system not running")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python soa_manager.py [start|stop|restart|status|service]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    project_root = Path(__file__).parent.parent
    manager = SOAManager(project_root)
    
    try:
        if command == "start":
            success = manager.start_all_services()
            sys.exit(0 if success else 1)
        elif command == "stop":
            manager.stop_all_services()
        elif command == "restart":
            manager.stop_all_services()
            time.sleep(2)
            success = manager.start_all_services()
            sys.exit(0 if success else 1)
        elif command == "status":
            manager.status()
        elif command == "service":
            if len(sys.argv) < 4:
                print("Usage: python soa_manager.py service [start|stop|restart] <service_name>")
                sys.exit(1)
            action = sys.argv[2]
            name = sys.argv[3]
            if action == 'start':
                ok = manager.start_single(name)
                sys.exit(0 if ok else 1)
            elif action == 'stop':
                ok = manager.stop_single(name)
                sys.exit(0 if ok else 1)
            elif action == 'restart':
                ok = manager.restart_single(name)
                sys.exit(0 if ok else 1)
            else:
                print(f"Unknown service action: {action}")
                sys.exit(1)
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        manager.stop_all_services()
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
