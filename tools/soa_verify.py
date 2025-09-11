#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from urllib.request import urlopen


def main() -> int:
    cfg = Path('soa_config.json')
    if not cfg.exists():
        print('soa_config.json not found; start SOA first', file=sys.stderr)
        return 1
    data = json.loads(cfg.read_text())
    hub_port = data['service_ports'].get('main_dashboard')
    if not hub_port:
        print('main_dashboard port not found in config', file=sys.stderr)
        return 1
    base = f'http://localhost:{hub_port}'
    def check(path: str) -> None:
        with urlopen(base + path, timeout=5) as r:
            print(f'  [OK] {path} -> {r.status}')
    for path in ['/soa_dashboard.html', '/api/soa-config', '/terminal', '/analytics', '/status', '/docs']:
        try:
            check(path)
        except Exception as e:
            print(f'  [FAIL] {path}: {e}', file=sys.stderr)
            return 1
    print('✅ SOA verify completed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

