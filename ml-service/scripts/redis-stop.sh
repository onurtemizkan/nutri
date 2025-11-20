#!/bin/bash

# Stop Redis server

set -e  # Exit on error

REDIS_PID_FILE="/tmp/nutri-redis.pid"
REDIS_PORT=6379

echo "🛑 Stopping Redis server..."

# Check if PID file exists
if [ ! -f "$REDIS_PID_FILE" ]; then
    # Try to find Redis by port
    if lsof -Pi :$REDIS_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        PID=$(lsof -Pi :$REDIS_PORT -sTCP:LISTEN -t)
        echo "⚠️  Found Redis running on port $REDIS_PORT (PID: $PID)"
        echo "   Attempting to stop..."
        kill -TERM "$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null
        sleep 1

        if ps -p "$PID" > /dev/null 2>&1; then
            echo "❌ Failed to stop Redis"
            exit 1
        else
            echo "✅ Redis stopped successfully"
            exit 0
        fi
    else
        echo "ℹ️  Redis is not running"
        exit 0
    fi
fi

# Stop Redis using PID file
PID=$(cat "$REDIS_PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "ℹ️  Redis is not running (stale PID file)"
    rm -f "$REDIS_PID_FILE"
    exit 0
fi

# Try graceful shutdown first
echo "📡 Sending SIGTERM to Redis (PID: $PID)..."
kill -TERM "$PID" 2>/dev/null || true

# Wait for graceful shutdown
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        rm -f "$REDIS_PID_FILE"
        echo "✅ Redis stopped successfully"
        exit 0
    fi
    sleep 0.5
done

# Force kill if still running
echo "⚠️  Forcing Redis shutdown..."
kill -KILL "$PID" 2>/dev/null || true
sleep 1

if ps -p "$PID" > /dev/null 2>&1; then
    echo "❌ Failed to stop Redis"
    exit 1
else
    rm -f "$REDIS_PID_FILE"
    echo "✅ Redis stopped (forced)"
    exit 0
fi
