#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
echo "🛑 Stopping HolographicMemory..."
docker compose down -v --remove-orphans || true
echo "✅ Stopped."

