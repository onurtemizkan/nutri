#!/bin/bash

# Check Redis server status and display information

REDIS_PID_FILE="/tmp/nutri-redis.pid"
REDIS_LOG_FILE="/tmp/nutri-redis.log"
REDIS_PORT=6379

echo "📊 Redis Status Check"
echo "===================="
echo ""

# Check if Redis command line tools are installed
if ! command -v redis-cli &> /dev/null; then
    echo "❌ Redis CLI not found"
    echo "📦 Install Redis: npm run setup:redis"
    exit 1
fi

# Check if Redis is running via PID file
if [ -f "$REDIS_PID_FILE" ]; then
    PID=$(cat "$REDIS_PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Redis is RUNNING"
        echo "   PID: $PID"
        echo "   Port: $REDIS_PORT"
        echo ""
    else
        echo "⚠️  Stale PID file found (Redis not running)"
        rm -f "$REDIS_PID_FILE"
    fi
else
    # Check if Redis is running on the port anyway
    if lsof -Pi :$REDIS_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        PID=$(lsof -Pi :$REDIS_PORT -sTCP:LISTEN -t)
        echo "⚠️  Redis is RUNNING (no PID file)"
        echo "   PID: $PID"
        echo "   Port: $REDIS_PORT"
        echo ""
    else
        echo "❌ Redis is NOT RUNNING"
        echo ""
        echo "🚀 Start Redis: npm run redis:start"
        exit 0
    fi
fi

# Try to ping Redis
if redis-cli -p $REDIS_PORT ping > /dev/null 2>&1; then
    echo "📡 Connection Status"
    echo "   Status: $(redis-cli -p $REDIS_PORT ping)"
    echo "   Port: $REDIS_PORT"
    echo ""

    # Get Redis info
    echo "📈 Server Information"
    redis-cli -p $REDIS_PORT INFO SERVER | grep -E "redis_version|os|tcp_port|uptime_in_seconds" | sed 's/^/   /'
    echo ""

    # Get memory info
    echo "💾 Memory Usage"
    redis-cli -p $REDIS_PORT INFO MEMORY | grep -E "used_memory_human|maxmemory_human" | sed 's/^/   /'
    echo ""

    # Get stats
    echo "📊 Statistics"
    redis-cli -p $REDIS_PORT INFO STATS | grep -E "total_connections_received|total_commands_processed|keyspace_hits|keyspace_misses" | sed 's/^/   /'
    echo ""

    # Get keyspace info
    echo "🔑 Keyspace"
    KEYSPACE=$(redis-cli -p $REDIS_PORT INFO KEYSPACE)
    if [ -z "$KEYSPACE" ]; then
        echo "   No keys stored"
    else
        echo "$KEYSPACE" | sed 's/^/   /'
    fi
    echo ""

    # Get number of keys
    KEY_COUNT=$(redis-cli -p $REDIS_PORT DBSIZE | awk '{print $2}')
    echo "   Total keys: $KEY_COUNT"
    echo ""

    # Show recent log entries
    if [ -f "$REDIS_LOG_FILE" ]; then
        echo "📝 Recent Logs (last 5 lines)"
        tail -5 "$REDIS_LOG_FILE" | sed 's/^/   /'
    fi

else
    echo "❌ Cannot connect to Redis on port $REDIS_PORT"
    echo ""
    if [ -f "$REDIS_LOG_FILE" ]; then
        echo "📝 Recent Logs"
        tail -10 "$REDIS_LOG_FILE" | sed 's/^/   /'
    fi
    exit 1
fi

echo ""
echo "🛑 To stop: npm run redis:stop"
echo "🗑️  To flush: npm run redis:flush"
echo "💻 To connect: npm run redis:cli"
