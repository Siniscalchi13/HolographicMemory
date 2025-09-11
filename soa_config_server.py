#!/usr/bin/env python3
"""
SOA Config Server
=================

Simple HTTP server that serves the SOA configuration for the main dashboard.
This allows the dashboard to discover actual port assignments.
"""

import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SOAConfigHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving SOA configuration"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/soa-config':
            self.serve_soa_config()
        elif self.path == '/soa_dashboard.html':
            self.serve_dashboard()
        elif self.path == '/sh-logo-square-dark.svg':
            self.serve_logo()
        elif self.path.startswith('/assets/images/sh-logo-dark.svg'):
            self.serve_assets_logo()
        elif self.path.startswith('/assets/images/smarthaus-logo-light.svg'):
            self.serve_smarthaus_logo()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        # Simple service control proxy -> invokes soa_manager.py
        if self.path.startswith('/api/services/'):
            try:
                _, _, service_segment = self.path.partition('/api/services/')
                parts = service_segment.split('/')
                if len(parts) != 2:
                    self.send_error(400, "Invalid service control path")
                    return
                service_name, action = parts[0], parts[1]
                from subprocess import run
                import sys
                cmd = [sys.executable, 'scripts/soa_manager.py', 'service', action, service_name]
                result = run(cmd, capture_output=True, text=True)
                status = 200 if result.returncode == 0 else 500
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                payload = {
                    'service': service_name,
                    'action': action,
                    'ok': result.returncode == 0,
                    'stdout': result.stdout[-1000:],
                    'stderr': result.stderr[-1000:],
                }
                self.wfile.write(json.dumps(payload).encode('utf-8'))
            except Exception as e:
                logger.error("Service control error: %s", e)
                self.send_error(500, "Internal Server Error")
        else:
            self.send_error(404, "Not Found")
    
    def serve_soa_config(self):
        """Serve the SOA configuration JSON with enhanced port registry"""
        try:
            config_file = Path("soa_config.json")
            registry_file = Path("soa_ports.json")
            
            # Load base config
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {"service_ports": {}, "timestamp": time.time()}
            
            # Load enhanced port registry if available
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                # Merge registry into config
                config["port_registry"] = registry
                config["navigation_urls"] = self._build_navigation_urls(registry)
            else:
                config["port_registry"] = None
                config["navigation_urls"] = {}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(config).encode('utf-8'))
            
        except Exception as e:
            logger.error("Error serving SOA config: %s", e)
            self.send_error(500, "Internal Server Error")
    
    def _build_navigation_urls(self, registry):
        """Build navigation URLs from registry for easy frontend consumption"""
        nav_urls = {}
        services = registry.get("services", {})
        
        # Main dashboard
        if "main_dashboard" in services:
            nav_urls["dashboard"] = services["main_dashboard"].get("url")
        
        # Terminal interface
        if "terminal_interface" in services:
            nav_urls["terminal"] = services["terminal_interface"].get("url")
        
        # Analytics dashboard
        if "analytics_dashboard" in services:
            nav_urls["analytics"] = services["analytics_dashboard"].get("url")
        
        # API status page
        if "api_status" in services:
            nav_urls["status"] = services["api_status"].get("url")
        
        # API docs
        if "api" in services:
            nav_urls["docs"] = services["api"].get("docs_url")
        
        return nav_urls
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        try:
            dashboard_file = Path("soa_dashboard.html")
            if dashboard_file.exists():
                with open(dashboard_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, "Dashboard not found")
                
        except Exception as e:
            logger.error("Error serving dashboard: %s", e)
            self.send_error(500, "Internal Server Error")
    
    def serve_logo(self):
        """Serve the SmartHaus logo"""
        try:
            logo_file = Path("sh-logo-square-dark.svg")
            if logo_file.exists():
                with open(logo_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', 'image/svg+xml')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, "Logo not found")
                
        except Exception as e:
            logger.error("Error serving logo: %s", e)
            self.send_error(500, "Internal Server Error")

    def serve_assets_logo(self):
        """Serve the horizontal SmartHaus logo from /assets/images"""
        try:
            logo_path = Path("assets/images/sh-logo-dark.svg")
            if logo_path.exists():
                with open(logo_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', 'image/svg+xml')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, "Logo not found")
        except Exception as e:
            logger.error("Error serving assets logo: %s", e)
            self.send_error(500, "Internal Server Error")

    def serve_smarthaus_logo(self):
        """Serve the SmartHaus logo from /assets/images"""
        try:
            logo_path = Path("assets/images/smarthaus-logo-light.svg")
            if logo_path.exists():
                with open(logo_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', 'image/svg+xml')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, "Logo not found")
        except Exception as e:
            logger.error("Error serving SmartHaus logo: %s", e)
            self.send_error(500, "Internal Server Error")
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(format % args)

def run_config_server(port=8080):
    """Run the SOA config server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, SOAConfigHandler)
    
    logger.info("🌐 SOA Config Server starting on port %d", port)
    logger.info("📊 Dashboard available at: http://localhost:%d/soa_dashboard.html", port)
    logger.info("🔧 Config API available at: http://localhost:%d/api/soa-config", port)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 SOA Config Server stopping...")
        httpd.shutdown()

if __name__ == '__main__':
    import sys
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port, using default 8080")
    
    run_config_server(port)
