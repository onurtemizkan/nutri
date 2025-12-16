"""
Redis client for caching features, predictions, and models
Uses async redis for non-blocking operations
"""

import json
import logging
from typing import Any, Optional
import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client wrapper with helper methods"""

    def __init__(self):
        self.redis: Optional[Redis] = None

    async def connect(self) -> None:
        """Initialize Redis connection"""
        try:
            self.redis = await aioredis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.redis_max_connections,
            )
            # Test connection
            await self.redis.ping()
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {str(e)}")
            self.redis = None

    async def close(self) -> None:
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.
        Automatically deserializes JSON.
        """
        if not self.redis:
            logger.warning("Redis not connected, skipping cache get")
            return None

        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in Redis cache.
        Automatically serializes to JSON.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            logger.warning("Redis not connected, skipping cache set")
            return False

        try:
            json_value = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, json_value)
            else:
                await self.redis.set(key, json_value)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self.redis:
            return False

        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error for key '{key}': {str(e)}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.redis:
            return False

        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key '{key}': {str(e)}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        Example: delete_pattern("features:user:123:*")

        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(
                f"Redis DELETE_PATTERN error for pattern '{pattern}': {str(e)}"
            )
            return 0

    async def get_ttl(self, key: str) -> Optional[int]:
        """Get time-to-live for a key in seconds"""
        if not self.redis:
            return None

        try:
            ttl = await self.redis.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Redis TTL error for key '{key}': {str(e)}")
            return None

    # Helper methods for ML-specific caching

    async def cache_features(
        self, user_id: str, date: str, category: str, features: dict
    ) -> bool:
        """Cache engineered features for a user and date"""
        key = f"features:{user_id}:{date}:{category}"
        return await self.set(key, features, ttl=settings.cache_ttl_features)

    async def get_features(
        self, user_id: str, date: str, category: str
    ) -> Optional[dict]:
        """Get cached features"""
        key = f"features:{user_id}:{date}:{category}"
        return await self.get(key)

    async def cache_prediction(
        self, user_id: str, metric_type: str, date: str, prediction: dict
    ) -> bool:
        """Cache ML prediction"""
        key = f"prediction:{user_id}:{metric_type}:{date}"
        return await self.set(key, prediction, ttl=settings.cache_ttl_predictions)

    async def get_prediction(
        self, user_id: str, metric_type: str, date: str
    ) -> Optional[dict]:
        """Get cached prediction"""
        key = f"prediction:{user_id}:{metric_type}:{date}"
        return await self.get(key)

    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache for a user (features + predictions)"""
        count = 0
        count += await self.delete_pattern(f"features:{user_id}:*")
        count += await self.delete_pattern(f"prediction:{user_id}:*")
        logger.info(f"Invalidated {count} cache keys for user {user_id}")
        return count


# Global Redis client instance
redis_client = RedisClient()


async def get_redis() -> RedisClient:
    """
    Dependency function to get Redis client.

    Usage in FastAPI:
        @app.get("/example")
        async def example(redis: RedisClient = Depends(get_redis)):
            await redis.set("key", "value")
    """
    return redis_client
