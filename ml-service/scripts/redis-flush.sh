#!/bin/bash

# Flush all Redis data (useful between test runs)

REDIS_PORT=6379

echo "🗑️  Flushing Redis data..."

# Check if Redis is running
if ! redis-cli -p $REDIS_PORT ping > /dev/null 2>&1; then
    echo "❌ Redis is not running"
    echo "🚀 Start Redis: npm run redis:start"
    exit 1
fi

# Get current key count
KEY_COUNT=$(redis-cli -p $REDIS_PORT DBSIZE | awk '{print $2}')
echo "📊 Current keys: $KEY_COUNT"

if [ "$KEY_COUNT" -eq 0 ]; then
    echo "✅ Redis is already empty"
    exit 0
fi

# Flush all databases
redis-cli -p $REDIS_PORT FLUSHALL > /dev/null

# Verify flush
NEW_COUNT=$(redis-cli -p $REDIS_PORT DBSIZE | awk '{print $2}')

if [ "$NEW_COUNT" -eq 0 ]; then
    echo "✅ Redis flushed successfully ($KEY_COUNT keys removed)"
else
    echo "⚠️  Redis flush may have failed (still $NEW_COUNT keys)"
    exit 1
fi
