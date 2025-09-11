# SOA Operations Runbook (Dev)

## Goals
- Deterministic start/stop/restart
- Single hub origin with proxy paths for all UIs
- Reliable cleanup to avoid stale ports/processes

## Commands
- Stop everything (force):
  - make soa-kill
- Fresh workspace:
  - make clean
- Setup environment:
  - make dev && source venv313/bin/activate
- Verify environment:
  - make dev-verify
- Start SOA:
  - make soa-start
- Restart SOA:
  - make soa-restart
- Stop SOA:
  - make soa-stop
- Status:
  - make soa-status
- Verify hub links (post-start):
  - make soa-verify

## Expected Ports
Dynamic; consult `make soa-status` or `soa_config.json`. The hub publishes navigation URLs that point to hub paths, independent of underlying ports.

## Health and Debugging
- Hub config API: `GET /api/soa-config` (on hub origin)
- Link verification: `make soa-verify`
- If links fail:
  1) `make soa-kill && make clean`
  2) `make soa-start`
  3) Reload browser (hard refresh)
- Logs: `logs/` directory if present; otherwise check terminal output for each started process

## Security
- Login is cookie-based at the hub; credentials are dev defaults
- Terminal requires `TERMINAL_AUTH_TOKEN` passed by prompt (stored client-side)
- For production, implement real auth and CSRF protections on login

