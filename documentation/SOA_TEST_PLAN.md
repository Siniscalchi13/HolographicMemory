# SOA Test Plan (Dev)

## Scope
Validates navigation, auth, service controls, and live telemetry in the single-origin hub architecture.

## Pre-requisites
- make dev && source venv313/bin/activate
- make soa-start (fresh run recommended: make soa-kill && make clean first)

## Automated Checks
- `make soa-verify` runs `tools/soa_verify.py` to validate:
  - /soa_dashboard.html (200)
  - /api/soa-config (200)
  - /terminal (200)
  - /analytics (200)
  - /status (200)
  - /docs (200)

## Manual E2E
1) Login Flow
- Open /soa_login.html → login with admin/soa2024 → redirected to /soa_dashboard.html
- Click logout (🚪) → redirected back to /soa_login.html

2) Navigation
- Header links: Dashboard/Terminal/Analytics/Status/Docs navigate without errors
- Body links mirror header behavior

3) Service Controls
- Restart API, Analytics, Status from cards → observe status, uptime, and latency update
- Stop/Start Terminal Service and confirm terminal connection reacts accordingly

4) Live Telemetry
- Header pills update CPU, Mem, Disk, Net, Conns in near real-time
- Analytics charts update over time; zoom/pan functional

5) Terminal WS
- Open Terminal; enter token when prompted
- Run basic commands (ls, pwd, cd, whoami)
- Resize window; terminal content reflows
- Stop terminal service; observe disconnect notice; restart and reconnect

6) Error Handling
- With hub down: endpoints return non-200; verify make soa-verify surfaces failures
- With stale processes: run make soa-kill && make clean and retry

## Acceptance Criteria
- All automated checks pass
- All navigation links work across pages
- Login/logout works consistently; access controlled
- Service controls perform the intended action; UI reflects state changes
- Metrics and charts update live; terminal is functional and secure

