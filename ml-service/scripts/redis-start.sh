#!/bin/bash

# Start Redis server for local development/testing

set -e  # Exit on error

REDIS_PID_FILE="/tmp/nutri-redis.pid"
REDIS_LOG_FILE="/tmp/nutri-redis.log"
REDIS_PORT=6379

echo "🚀 Starting Redis server..."

# Check if Redis is already running
if [ -f "$REDIS_PID_FILE" ]; then
    PID=$(cat "$REDIS_PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Redis is already running (PID: $PID)"
        echo "📊 Check status: npm run redis:status"
        exit 0
    else
        # Stale PID file, remove it
        rm -f "$REDIS_PID_FILE"
    fi
fi

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "❌ Redis is not installed!"
    echo "📦 Install Redis: npm run setup:redis"
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$REDIS_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $REDIS_PORT is already in use"
    echo "   Another Redis instance might be running"
    echo "   Use 'npm run redis:stop' to stop it or check with 'npm run redis:status'"
    exit 1
fi

# Start Redis in background
echo "📡 Starting Redis on port $REDIS_PORT..."
redis-server \
    --daemonize yes \
    --port $REDIS_PORT \
    --pidfile "$REDIS_PID_FILE" \
    --logfile "$REDIS_LOG_FILE" \
    --dir /tmp \
    --save "" \
    --appendonly no

# Wait for Redis to start
echo "⏳ Waiting for Redis to be ready..."
for i in {1..10}; do
    if redis-cli -p $REDIS_PORT ping > /dev/null 2>&1; then
        echo "✅ Redis started successfully!"
        echo "   PID: $(cat $REDIS_PID_FILE)"
        echo "   Port: $REDIS_PORT"
        echo "   Logs: $REDIS_LOG_FILE"
        echo ""
        echo "🛑 To stop: npm run redis:stop"
        echo "📊 To check status: npm run redis:status"
        echo "🗑️  To flush data: npm run redis:flush"
        exit 0
    fi
    sleep 0.5
done

echo "❌ Redis failed to start within timeout"
echo "📝 Check logs: cat $REDIS_LOG_FILE"
exit 1
