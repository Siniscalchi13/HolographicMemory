#!/bin/bash

# HolographicMemory Clean Launch Script
# Kills existing processes and starts frontend + backend in correct order

echo "🚀 Starting HolographicMemory Clean Launch..."

# Function to find available port
find_available_port() {
    local port=$1
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        echo "Port $port is in use, trying $((port + 1))"
        port=$((port + 1))
    done
    echo $port
}

# Kill existing processes
echo "🧹 Cleaning up existing processes..."

# Kill API processes on common ports
for port in 8000 8001 8080 3000; do
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "Killing process on port $port"
        lsof -ti:$port | xargs kill -9 2>/dev/null
    fi
done

# Kill Electron processes
pkill -f "electron.*holo-desktop" 2>/dev/null && echo "Killed existing Electron processes" || echo "No Electron processes to kill"

# Kill uvicorn processes
pkill -f "uvicorn.*app:app" 2>/dev/null && echo "Killed existing API processes" || echo "No API processes to kill"

sleep 2

# Find available ports
API_PORT=$(find_available_port 8000)
echo "✅ Using API port: $API_PORT"

# Start API backend
echo "🔧 Starting API backend on port $API_PORT..."
cd services/api
uvicorn app:app --reload --host 0.0.0.0 --port $API_PORT &
API_PID=$!

# Wait for API to start
echo "⏳ Waiting for API to initialize..."
sleep 5

# Test API
if curl -s http://localhost:$API_PORT/healthz >/dev/null 2>&1; then
    echo "✅ API is running on port $API_PORT"
else
    echo "❌ API failed to start"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Update desktop app config if needed
cd ../../desktop/holo-desktop

# Check if we need to update the API URL in the app
if [ "$API_PORT" != "8000" ]; then
    echo "📝 Updating desktop app to use port $API_PORT"
    # You might need to update a config file here if the app hardcodes port 8000
fi

# Start desktop app
echo "🖥️  Starting desktop application..."
npm start &
ELECTRON_PID=$!

echo ""
echo "🎉 HolographicMemory is now running!"
echo "📡 API: http://localhost:$API_PORT"
echo "🖥️  Desktop: Opening automatically"
echo ""
echo "To stop everything, press Ctrl+C or run: pkill -f 'uvicorn|electron.*holo-desktop'"
echo ""

# Wait for user to stop
trap "echo '🛑 Stopping HolographicMemory...'; kill $API_PID $ELECTRON_PID 2>/dev/null; exit 0" INT

# Keep script running
wait
