#!/bin/bash

# ONE FUCKING COMMAND TO RULE THEM ALL

# Handle stop command
if [ "$1" = "stop" ]; then
    echo "🛑 Stopping HolographicMemory..."
    docker compose down -v
    echo "✅ Stopped"
    exit 0
fi

echo "🚀 Starting HolographicMemory..."

# Clean up any existing shit
docker compose down -v 2>/dev/null

# Start everything
docker compose up -d --build --remove-orphans

# Wait for it to be ready
echo "⏳ Waiting for system..."
for i in {1..30}; do
    if curl -s http://localhost:8000/healthz >/dev/null 2>&1; then
        echo ""
        echo "✅ READY!"
        echo ""
        echo "👉 Open this in your browser: http://localhost:8000"
        echo ""
        echo "To stop: ./launch_holo.sh stop"
        echo ""
        
        # Try to open browser automatically
        if command -v open &> /dev/null; then
            open http://localhost:8000
        fi
        exit 0
    fi
    sleep 1
done

echo "❌ Failed to start. Check Docker is running."
exit 1
