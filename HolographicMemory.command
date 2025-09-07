#!/bin/bash
# Double-click launcher for HolographicMemory (macOS)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "🚀 Starting HolographicMemory..."

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker is not installed. Please install Docker Desktop for Mac."
  exit 1
fi

echo "🐳 Bringing up containers..."
docker compose up -d --remove-orphans

echo "⏳ Waiting for API at http://localhost:8000..."
for i in {1..45}; do
  if curl -sSf http://localhost:8000/healthz >/dev/null; then
    echo "✅ API is running"
    break
  fi
  if [ $i -eq 45 ]; then
    echo "❌ API failed to start. Showing logs:"
    docker compose logs --tail=200 api
    exit 1
  fi
  sleep 1
done

echo "🌐 Opening web app..."
open http://localhost:8000/

echo "📌 HolographicMemory is running. Close this window to stop logs."
echo "📝 Tip: Create an alias to this script in Applications for easy launching."

# Follow logs until user closes
docker compose logs -f api

