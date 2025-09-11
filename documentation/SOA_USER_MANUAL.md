# SmartHaus SOA Dashboard — User Manual

## Overview
The SmartHaus SOA Dashboard is the single entry point for operating the HolographicMemory system. It provides login‑gated access to:
- Main Dashboard: Status, live metrics, service controls
- Terminal: Real shell access in the browser (token‑secured)
- Analytics: Live charts and performance telemetry
- Status: API status/overview
- API Docs: Interactive OpenAPI docs

## Accessing the System
1. Start the system (developer workflow):
   - make soa-kill && make clean
   - make dev && source venv313/bin/activate
   - make dev-verify
   - make soa-start
2. Open the login page shown in the console (e.g. http://localhost:8100/soa_login.html)
3. Sign in (dev credentials):
   - Username: admin
   - Password: soa2024

## Navigation
The header provides consistent links across all pages:
- Dashboard: /soa_dashboard.html
- Terminal: /terminal
- Analytics: /analytics
- Status: /status
- Docs: /docs

These paths are on the same origin (the hub), ensuring stable auth and consistent styling.

## Main Dashboard
- Live Metrics: CPU, Memory, Disk, Network, Connections, GPU (placeholder until GPU provider configured)
- Service Cards: Running/Stopped, Uptime, measured Latency, and quick actions (Start/Stop/Restart)
- Refresh: Use the refresh button (bottom‑right) to recheck status
- Logout: Click the logout button (🚪) to end your session

## Terminal
- Launch from header: Terminal → /terminal
- Authentication: Terminal requires a token; when prompted, enter the value of TERMINAL_AUTH_TOKEN used by the terminal service
- Resize: The terminal resizes with the window
- Usage: Execute shell commands (ls, pwd, cd, etc.)
- Disconnect/Reconnect: If the terminal service is restarted, the session will disconnect and can be reconnected by refreshing the page

## Analytics
- Live charts for CPU, Memory, GPU% (placeholder), and Network
- Zoom/pan supported (wheel or pinch)
- Auto‑updates via WebSocket stream

## Status Page
- Static overview of API and endpoints

## Service Controls
On the main dashboard cards:
- Restart API / Analytics / Status
- Start/Stop Terminal Service
- Controls trigger server‑side actions; status and uptime update after a moment

## Logging Out
- Click the logout floating button (🚪) on the main dashboard header section
- This clears your session and returns you to the login page

## Troubleshooting
- If links don’t work, run: make soa-kill && make clean && make soa-start
- Verify hub endpoints: source venv313/bin/activate && make soa-verify
- Terminal token: Ensure TERMINAL_AUTH_TOKEN is set in the terminal service environment
- Browser cache: Hard refresh after updates (Cmd+Shift+R / Ctrl+F5)

