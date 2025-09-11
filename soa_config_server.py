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
from urllib.parse import urlparse
import requests
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
        elif self.path == '/soa_login.html':
            self.serve_login()
        elif self.path == '/sh-logo-square-dark.svg':
            self.serve_logo()
        elif self.path.startswith('/assets/images/sh-logo-dark.svg'):
            self.serve_sh_logo_dark()
        elif self.path.startswith('/assets/images/smarthaus-logo-light.svg'):
            self.serve_smarthaus_logo()
        elif self.path == '/terminal':
            if not self._is_authenticated():
                return self._redirect('/soa_login.html')
            self.proxy_service('terminal_interface', '/')
        elif self.path == '/analytics':
            if not self._is_authenticated():
                return self._redirect('/soa_login.html')
            self.proxy_service('analytics_dashboard', '/')
        elif self.path == '/status':
            if not self._is_authenticated():
                return self._redirect('/soa_login.html')
            self.proxy_service('api_status', '/')
        elif self.path.startswith('/docs'):
            if not self._is_authenticated():
                return self._redirect('/soa_login.html')
            # Proxy /docs and any subpaths to API service
            self.proxy_service('api', self.path)
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
        elif self.path == '/login':
            # Trivial login handler (dev). In prod, replace with real auth.
            length = int(self.headers.get('Content-Length', 0) or 0)
            body = self.rfile.read(length).decode('utf-8') if length else ''
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}
            username = data.get('username')
            password = data.get('password')
            if username == 'admin' and password == 'soa2024':
                # Set cookie and redirect
                self.send_response(200)
                self.send_header('Set-Cookie', 'soa_auth=1; Path=/; HttpOnly; SameSite=Lax')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True}).encode('utf-8'))
            else:
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': 'Invalid credentials'}).encode('utf-8'))
        elif self.path == '/logout':
            self.send_response(200)
            self.send_header('Set-Cookie', 'soa_auth=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True}).encode('utf-8'))
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
        services = registry.get("services", {})
        # Build hub base from Host header
        host = self.headers.get('Host') or f"localhost:{getattr(self.server, 'server_port', 8080)}"
        base = f"http://{host}"
        return {
            'dashboard': f"{base}/soa_dashboard.html",
            'terminal' : f"{base}/terminal",
            'analytics': f"{base}/analytics",
            'status'   : f"{base}/status",
            'docs'     : f"{base}/docs",
        }
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        try:
            if not self._is_authenticated():
                return self._redirect('/soa_login.html')
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
    
    def serve_login(self):
        """Serve the login page HTML"""
        try:
            login_file = Path("soa_login.html")
            if login_file.exists():
                with open(login_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, "Login page not found")
                
        except Exception as e:
            logger.error("Error serving login page: %s", e)
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
    
    def serve_sh_logo_dark(self):
        """Serve the SmartHaus dark logo from /assets/images"""
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
            logger.error("Error serving SmartHaus dark logo: %s", e)
            self.send_error(500, "Internal Server Error")
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(format % args)

    # --- Helpers ---
    def _is_authenticated(self) -> bool:
        cookies = self.headers.get('Cookie', '')
        return 'soa_auth=1' in cookies

    def _redirect(self, path: str):
        self.send_response(302)
        self.send_header('Location', path)
        self.end_headers()

    def _get_service_port(self, name: str) -> int | None:
        try:
            with open('soa_config.json', 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            return cfg.get('service_ports', {}).get(name)
        except Exception:
            return None

    def proxy_service(self, service_name: str, upstream_path: str):
        port = self._get_service_port(service_name)
        if not port:
            self.send_error(502, f"Service {service_name} not available")
            return
        url = f"http://localhost:{port}{upstream_path}"
        try:
            # Pass through simple GET proxy
            resp = requests.get(url, timeout=5)
            self.send_response(resp.status_code)
            # Basic content-type passthrough
            ctype = resp.headers.get('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Type', ctype)
            self.end_headers()
            self.wfile.write(resp.content)
        except requests.RequestException as e:
            logger.error("Proxy error for %s to %s: %s", service_name, url, e)
            self.send_error(502, "Bad Gateway")

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
