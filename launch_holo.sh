#!/bin/bash

# HolographicMemory Docker Launch Script
# Uses Docker for API, serves web UI - no global Python dependencies

echo "🚀 Starting HolographicMemory (Docker + Web UI)..."

# Kill existing processes
echo "🧹 Cleaning up existing processes..."

# Stop any existing Docker containers
docker compose down 2>/dev/null

sleep 2

# Start Docker API
echo "🐳 Starting API in Docker..."
docker compose up -d --remove-orphans

# Wait for API to be healthy
echo "⏳ Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/healthz >/dev/null 2>&1; then
        echo "✅ API is running at http://localhost:8000"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API failed to start after 30 seconds"
        docker compose logs api
        exit 1
    fi
    sleep 1
done

echo ""
echo "🎉 HolographicMemory is now running!"
echo "📡 API: http://localhost:8000 (Docker)"
echo "🌐 Web UI: http://localhost:8000"
echo ""
echo "To stop:"
echo "  make down (stops Docker API)"
echo ""
echo "Open your browser to http://localhost:8000 to use the web interface!"
