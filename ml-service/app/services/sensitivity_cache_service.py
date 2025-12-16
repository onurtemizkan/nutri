"""
High-Performance Redis Caching Layer for Sensitivity Detection.

This module provides a sophisticated caching infrastructure with:
- Multi-tier caching (L1 in-memory + L2 Redis)
- Intelligent cache warming and prefetching
- User-partitioned cache namespaces
- Bloom filter for negative caching
- Write-through and write-behind strategies
- Cache statistics and monitoring

Performance targets:
- Cache hit latency: <1ms (L1), <5ms (L2)
- Cache miss with DB fetch: <50ms
- Memory efficiency: ~70% hit rate target
"""
# mypy: ignore-errors

import asyncio
import hashlib
import pickle
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)
import math

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CachePolicy(Enum):
    """Cache eviction and update policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time-To-Live based
    ADAPTIVE = "adaptive"  # Combines LRU + LFU based on access patterns


class CacheTier(Enum):
    """Cache tier levels."""

    L1_MEMORY = "l1"  # In-process memory cache
    L2_REDIS = "l2"  # Redis distributed cache
    L3_COMPRESSED = "l3"  # Compressed cold storage


@dataclass
class CacheEntry(Generic[T]):
    """Individual cache entry with metadata."""

    key: str
    value: T
    created_at: float
    accessed_at: float
    access_count: int
    ttl_seconds: int
    size_bytes: int
    tier: CacheTier
    compressed: bool = False
    checksum: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if entry has exceeded TTL."""
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    evictions: int = 0
    expirations: int = 0
    writes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 cache hit rate."""
        return self.l1_hits / self.hits if self.hits > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "l1_hit_rate": f"{self.l1_hit_rate:.2%}",
            "evictions": self.evictions,
            "expirations": self.expirations,
            "writes": self.writes,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "avg_hit_latency_ms": f"{self.avg_hit_latency_ms:.2f}",
            "avg_miss_latency_ms": f"{self.avg_miss_latency_ms:.2f}",
        }


class BloomFilter:
    """
    Space-efficient probabilistic data structure for negative caching.

    Used to quickly determine if a key definitely does NOT exist in cache,
    avoiding unnecessary cache lookups for non-existent keys.

    False positive rate ≈ (1 - e^(-kn/m))^k
    where k = number of hash functions, n = number of elements, m = bit array size
    """

    def __init__(
        self, expected_elements: int = 100000, false_positive_rate: float = 0.01
    ):
        """
        Initialize Bloom filter.

        Args:
            expected_elements: Expected number of elements
            false_positive_rate: Desired false positive rate
        """
        # Calculate optimal size and hash count
        # m = -n * ln(p) / (ln(2))^2
        self.size = int(
            -expected_elements * math.log(false_positive_rate) / (math.log(2) ** 2)
        )
        # k = (m/n) * ln(2)
        self.hash_count = int((self.size / expected_elements) * math.log(2))
        self.hash_count = max(1, self.hash_count)

        # Bit array (using bytearray for efficiency)
        self.bit_array = bytearray((self.size + 7) // 8)
        self.element_count = 0

    def _get_hash_values(self, key: str) -> List[int]:
        """Generate k hash values using double hashing."""
        # Use two hash functions to generate k hash values
        h1 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(key.encode()).hexdigest(), 16)

        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def add(self, key: str) -> None:
        """Add a key to the Bloom filter."""
        for pos in self._get_hash_values(key):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self.bit_array[byte_idx] |= 1 << bit_idx
        self.element_count += 1

    def might_contain(self, key: str) -> bool:
        """
        Check if key might be in the filter.

        Returns:
            False: Key is definitely NOT in the set
            True: Key MIGHT be in the set (could be false positive)
        """
        for pos in self._get_hash_values(key):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def estimated_false_positive_rate(self) -> float:
        """Calculate current estimated false positive rate."""
        # p ≈ (1 - e^(-kn/m))^k
        exponent = -self.hash_count * self.element_count / self.size
        return (1 - math.exp(exponent)) ** self.hash_count


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache for L1 in-memory caching.

    Uses OrderedDict for O(1) access and eviction.
    """

    def __init__(self, max_size: int = 10000, max_memory_mb: int = 100):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._current_memory = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get entry from cache, updating access order."""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._current_memory -= entry.size_bytes
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            return entry

    async def set(self, key: str, value: T, ttl_seconds: int = 3600) -> CacheEntry[T]:
        """Set entry in cache with eviction if needed."""
        async with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default estimate

            # Evict if needed
            while (
                len(self._cache) >= self.max_size
                or self._current_memory + size_bytes > self.max_memory_bytes
            ):
                if not self._cache:
                    break
                # Remove oldest (first) item
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._current_memory -= oldest_entry.size_bytes

            # Create entry
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                access_count=1,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                tier=CacheTier.L1_MEMORY,
                checksum=hashlib.md5(str(value).encode()).hexdigest()[:8],
            )

            self._cache[key] = entry
            self._current_memory += size_bytes

            return entry

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._current_memory -= entry.size_bytes
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries from cache."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_memory = 0
            return count

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    @property
    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self._current_memory / (1024 * 1024)


class SensitivityCacheService:
    """
    Multi-tier caching service for sensitivity detection.

    Architecture:
    - L1: In-memory LRU cache (fastest, limited size)
    - L2: Redis distributed cache (fast, large capacity)
    - L3: Compressed Redis storage (cold data, highest capacity)

    Features:
    - Automatic tier promotion/demotion
    - Bloom filter for negative caching
    - Write-through with async write-behind
    - Cache warming and prefetching
    - Namespace isolation per user
    """

    # Cache key prefixes
    PREFIX_INGREDIENT = "ing:"
    PREFIX_SENSITIVITY = "sens:"
    PREFIX_USER = "user:"
    PREFIX_ANALYSIS = "analysis:"
    PREFIX_MODEL = "model:"

    # Default TTLs (seconds)
    TTL_INGREDIENT = 86400  # 24 hours
    TTL_SENSITIVITY = 3600  # 1 hour
    TTL_ANALYSIS = 1800  # 30 minutes
    TTL_MODEL = 43200  # 12 hours

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        l1_max_size: int = 10000,
        l1_max_memory_mb: int = 100,
        enable_bloom_filter: bool = True,
        enable_compression: bool = True,
        compression_threshold_bytes: int = 1024,
    ):
        """
        Initialize cache service.

        Args:
            redis_url: Redis connection URL
            l1_max_size: Maximum L1 cache entries
            l1_max_memory_mb: Maximum L1 memory in MB
            enable_bloom_filter: Enable Bloom filter for negative caching
            enable_compression: Enable compression for large values
            compression_threshold_bytes: Minimum size for compression
        """
        self.redis_url = redis_url
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold_bytes

        # L1 in-memory cache
        self.l1_cache: LRUCache[Any] = LRUCache(
            max_size=l1_max_size, max_memory_mb=l1_max_memory_mb
        )

        # L2 Redis cache (initialized lazily)
        self._redis: Optional[Any] = None
        self._redis_available = False

        # Bloom filter for negative caching
        self.bloom_filter = BloomFilter() if enable_bloom_filter else None

        # Statistics
        self.stats = CacheStats()

        # Write-behind queue
        self._write_queue: asyncio.Queue[Tuple[str, Any, int]] = asyncio.Queue()
        self._write_task: Optional[asyncio.Task] = None

        # Prefetch queue
        self._prefetch_queue: asyncio.Queue[List[str]] = asyncio.Queue()
        self._prefetch_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """Initialize Redis connection and background tasks."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, using L1 cache only")
            return False

        try:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle encoding ourselves
            )
            await self._redis.ping()
            self._redis_available = True
            logger.info("Redis connection established")

            # Start background tasks
            self._write_task = asyncio.create_task(self._write_behind_worker())
            self._prefetch_task = asyncio.create_task(self._prefetch_worker())

            return True

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using L1 cache only")
            self._redis_available = False
            return False

    async def close(self) -> None:
        """Close connections and stop background tasks."""
        if self._write_task:
            self._write_task.cancel()
        if self._prefetch_task:
            self._prefetch_task.cancel()
        if self._redis:
            await self._redis.close()

    def _make_key(self, prefix: str, *parts: str) -> str:
        """Generate cache key from prefix and parts."""
        return prefix + ":".join(str(p) for p in parts)

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value)

        # Compress if enabled and above threshold
        if self.enable_compression and len(data) > self.compression_threshold:
            compressed = zlib.compress(data, level=6)
            # Only use compression if it actually reduces size
            if len(compressed) < len(data) * 0.9:
                return b"Z" + compressed  # Prefix to indicate compression

        return b"R" + data  # Raw data prefix

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize stored value."""
        if not data:
            return None

        prefix = data[0:1]
        payload = data[1:]

        if prefix == b"Z":
            payload = zlib.decompress(payload)

        return pickle.loads(payload)

    async def get(
        self,
        key: str,
        fetch_func: Optional[Callable[[], Any]] = None,
        ttl_seconds: int = 3600,
    ) -> Optional[Any]:
        """
        Get value from cache with optional cache-aside pattern.

        Args:
            key: Cache key
            fetch_func: Optional function to fetch value on cache miss
            ttl_seconds: TTL for fetched values

        Returns:
            Cached or fetched value, or None
        """
        start_time = time.time()

        # Check Bloom filter first (negative cache)
        if self.bloom_filter and not self.bloom_filter.might_contain(key):
            # Definitely not in cache
            if fetch_func:
                value = await self._fetch_and_cache(key, fetch_func, ttl_seconds)
                return value
            return None

        # Try L1 cache first
        l1_entry = await self.l1_cache.get(key)
        if l1_entry is not None:
            self.stats.hits += 1
            self.stats.l1_hits += 1
            self.stats.bytes_read += l1_entry.size_bytes
            self._update_hit_latency(start_time)
            return l1_entry.value

        # Try L2 Redis cache
        if self._redis_available:
            try:
                data = await self._redis.get(key)
                if data is not None:
                    value = self._deserialize(data)

                    # Promote to L1
                    await self.l1_cache.set(key, value, ttl_seconds)

                    self.stats.hits += 1
                    self.stats.l2_hits += 1
                    self.stats.bytes_read += len(data)
                    self._update_hit_latency(start_time)
                    return value
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        # Cache miss
        self.stats.misses += 1
        self._update_miss_latency(start_time)

        # Fetch if function provided
        if fetch_func:
            value = await self._fetch_and_cache(key, fetch_func, ttl_seconds)
            return value

        return None

    async def _fetch_and_cache(
        self, key: str, fetch_func: Callable[[], Any], ttl_seconds: int
    ) -> Any:
        """Fetch value and cache it."""
        # Handle both sync and async fetch functions
        if asyncio.iscoroutinefunction(fetch_func):
            value = await fetch_func()
        else:
            value = fetch_func()

        if value is not None:
            await self.set(key, value, ttl_seconds)

        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
        write_through: bool = True,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            write_through: If True, write to L2 synchronously

        Returns:
            True if successful
        """
        # Write to L1
        await self.l1_cache.set(key, value, ttl_seconds)

        # Update Bloom filter
        if self.bloom_filter:
            self.bloom_filter.add(key)

        # Write to L2
        if self._redis_available:
            if write_through:
                try:
                    data = self._serialize(value)
                    await self._redis.setex(key, ttl_seconds, data)
                    self.stats.writes += 1
                    self.stats.bytes_written += len(data)
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
                    return False
            else:
                # Queue for write-behind
                await self._write_queue.put((key, value, ttl_seconds))

        return True

    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers."""
        # Delete from L1
        await self.l1_cache.delete(key)

        # Delete from L2
        if self._redis_available:
            try:
                await self._redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                return False

        return True

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern (L2 only)."""
        if not self._redis_available:
            return 0

        try:
            count = 0
            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
                count += 1
            return count
        except Exception as e:
            logger.warning(f"Redis delete pattern error: {e}")
            return 0

    # ==================== Ingredient Cache ====================

    async def get_ingredient(
        self, ingredient_name: str, fetch_func: Optional[Callable[[], Any]] = None
    ) -> Optional[Any]:
        """Get ingredient data from cache."""
        key = self._make_key(self.PREFIX_INGREDIENT, ingredient_name.lower())
        return await self.get(key, fetch_func, self.TTL_INGREDIENT)

    async def set_ingredient(self, ingredient_name: str, data: Any) -> bool:
        """Cache ingredient data."""
        key = self._make_key(self.PREFIX_INGREDIENT, ingredient_name.lower())
        return await self.set(key, data, self.TTL_INGREDIENT)

    async def get_ingredient_batch(self, ingredient_names: List[str]) -> Dict[str, Any]:
        """Get multiple ingredients from cache."""
        results = {}
        missing = []

        for name in ingredient_names:
            value = await self.get_ingredient(name)
            if value is not None:
                results[name] = value
            else:
                missing.append(name)

        # Queue missing items for prefetch
        if missing:
            await self._prefetch_queue.put(
                [self._make_key(self.PREFIX_INGREDIENT, n.lower()) for n in missing]
            )

        return results

    # ==================== Sensitivity Cache ====================

    async def get_sensitivity(
        self,
        user_id: str,
        trigger_type: str,
        trigger_name: str = "",
        fetch_func: Optional[Callable[[], Any]] = None,
    ) -> Optional[Any]:
        """Get user sensitivity data from cache."""
        key = self._make_key(
            self.PREFIX_SENSITIVITY,
            user_id,
            trigger_type,
            trigger_name.lower() if trigger_name else "general",
        )
        return await self.get(key, fetch_func, self.TTL_SENSITIVITY)

    async def set_sensitivity(
        self, user_id: str, trigger_type: str, trigger_name: str, data: Any
    ) -> bool:
        """Cache user sensitivity data."""
        key = self._make_key(
            self.PREFIX_SENSITIVITY,
            user_id,
            trigger_type,
            trigger_name.lower() if trigger_name else "general",
        )
        return await self.set(key, data, self.TTL_SENSITIVITY)

    async def invalidate_user_sensitivity(self, user_id: str) -> int:
        """Invalidate all sensitivity cache for a user."""
        pattern = f"{self.PREFIX_SENSITIVITY}{user_id}:*"
        return await self.delete_pattern(pattern)

    # ==================== Analysis Cache ====================

    async def get_analysis(
        self,
        user_id: str,
        analysis_type: str,
        params_hash: str,
        fetch_func: Optional[Callable[[], Any]] = None,
    ) -> Optional[Any]:
        """Get analysis results from cache."""
        key = self._make_key(self.PREFIX_ANALYSIS, user_id, analysis_type, params_hash)
        return await self.get(key, fetch_func, self.TTL_ANALYSIS)

    async def set_analysis(
        self, user_id: str, analysis_type: str, params_hash: str, data: Any
    ) -> bool:
        """Cache analysis results."""
        key = self._make_key(self.PREFIX_ANALYSIS, user_id, analysis_type, params_hash)
        return await self.set(key, data, self.TTL_ANALYSIS)

    # ==================== Model Cache ====================

    async def get_model(
        self,
        model_name: str,
        version: str,
        fetch_func: Optional[Callable[[], Any]] = None,
    ) -> Optional[Any]:
        """Get ML model from cache."""
        key = self._make_key(self.PREFIX_MODEL, model_name, version)
        return await self.get(key, fetch_func, self.TTL_MODEL)

    async def set_model(self, model_name: str, version: str, data: Any) -> bool:
        """Cache ML model."""
        key = self._make_key(self.PREFIX_MODEL, model_name, version)
        return await self.set(key, data, self.TTL_MODEL)

    # ==================== Cache Warming ====================

    async def warm_ingredient_cache(self, ingredients: List[Dict[str, Any]]) -> int:
        """Pre-populate ingredient cache."""
        count = 0
        for ing in ingredients:
            name = ing.get("name", "")
            if name:
                await self.set_ingredient(name, ing)
                count += 1

        logger.info(f"Warmed ingredient cache with {count} entries")
        return count

    async def warm_user_cache(
        self, user_id: str, sensitivities: List[Dict[str, Any]]
    ) -> int:
        """Pre-populate user sensitivity cache."""
        count = 0
        for sens in sensitivities:
            trigger_type = sens.get("trigger_type", "")
            trigger_name = sens.get("trigger_name", "")
            if trigger_type:
                await self.set_sensitivity(user_id, trigger_type, trigger_name, sens)
                count += 1

        logger.info(f"Warmed user {user_id} cache with {count} entries")
        return count

    # ==================== Background Workers ====================

    async def _write_behind_worker(self) -> None:
        """Background worker for async writes to Redis."""
        batch: List[Tuple[str, Any, int]] = []
        batch_size = 100
        flush_interval = 1.0  # seconds
        last_flush = time.time()

        while True:
            try:
                # Collect items with timeout
                try:
                    item = await asyncio.wait_for(
                        self._write_queue.get(), timeout=flush_interval
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass

                # Flush batch if full or timeout
                should_flush = len(batch) >= batch_size or (
                    batch and time.time() - last_flush >= flush_interval
                )

                if should_flush and batch and self._redis_available:
                    pipe = self._redis.pipeline()
                    for key, value, ttl in batch:
                        data = self._serialize(value)
                        pipe.setex(key, ttl, data)
                        self.stats.bytes_written += len(data)

                    await pipe.execute()
                    self.stats.writes += len(batch)
                    batch.clear()
                    last_flush = time.time()

            except asyncio.CancelledError:
                # Flush remaining on shutdown
                if batch and self._redis_available:
                    pipe = self._redis.pipeline()
                    for key, value, ttl in batch:
                        data = self._serialize(value)
                        pipe.setex(key, ttl, data)
                    await pipe.execute()
                break
            except Exception as e:
                logger.error(f"Write-behind worker error: {e}")
                await asyncio.sleep(1.0)

    async def _prefetch_worker(self) -> None:
        """Background worker for cache prefetching."""
        while True:
            try:
                keys = await self._prefetch_queue.get()

                if not self._redis_available or not keys:
                    continue

                # Batch get from Redis
                values = await self._redis.mget(keys)

                # Promote to L1
                for key, data in zip(keys, values):
                    if data is not None:
                        value = self._deserialize(data)
                        await self.l1_cache.set(key, value, self.TTL_INGREDIENT)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                await asyncio.sleep(1.0)

    # ==================== Statistics ====================

    def _update_hit_latency(self, start_time: float) -> None:
        """Update average hit latency."""
        latency_ms = (time.time() - start_time) * 1000
        # Exponential moving average
        alpha = 0.1
        self.stats.avg_hit_latency_ms = (
            alpha * latency_ms + (1 - alpha) * self.stats.avg_hit_latency_ms
        )

    def _update_miss_latency(self, start_time: float) -> None:
        """Update average miss latency."""
        latency_ms = (time.time() - start_time) * 1000
        alpha = 0.1
        self.stats.avg_miss_latency_ms = (
            alpha * latency_ms + (1 - alpha) * self.stats.avg_miss_latency_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.stats.to_dict()
        stats.update(
            {
                "l1_size": self.l1_cache.size,
                "l1_memory_mb": f"{self.l1_cache.memory_usage_mb:.2f}",
                "redis_available": self._redis_available,
            }
        )

        if self.bloom_filter:
            stats["bloom_filter"] = {
                "elements": self.bloom_filter.element_count,
                "estimated_fpr": f"{self.bloom_filter.estimated_false_positive_rate():.4%}",
            }

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health status."""
        health = {
            "status": "healthy",
            "l1_cache": "ok",
            "l2_redis": "unavailable",
            "issues": [],
        }

        # Check L1
        if self.l1_cache.size == 0:
            health["issues"].append("L1 cache is empty")

        # Check L2
        if self._redis_available:
            try:
                await self._redis.ping()
                health["l2_redis"] = "ok"
            except Exception as e:
                health["l2_redis"] = f"error: {e}"
                health["issues"].append(f"Redis error: {e}")

        # Check hit rate
        if self.stats.hit_rate < 0.5 and self.stats.hits + self.stats.misses > 100:
            health["issues"].append(f"Low hit rate: {self.stats.hit_rate:.1%}")

        if health["issues"]:
            health["status"] = "degraded"

        return health


# ==================== Singleton Instance ====================

_cache_service: Optional[SensitivityCacheService] = None


async def get_cache_service() -> SensitivityCacheService:
    """Get or create the cache service singleton."""
    global _cache_service

    if _cache_service is None:
        _cache_service = SensitivityCacheService()
        await _cache_service.initialize()

    return _cache_service


# ==================== Cache Decorators ====================


def cached(
    prefix: str,
    ttl_seconds: int = 3600,
    key_builder: Optional[Callable[..., str]] = None,
):
    """
    Decorator for caching function results.

    Usage:
        @cached("analysis", ttl_seconds=1800)
        async def analyze_hrv(user_id: str, data: dict) -> dict:
            ...
    """

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            cache = await get_cache_service()

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default: hash arguments
                key_parts = [str(a) for a in args]
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            full_key = f"{prefix}:{cache_key}"

            # Try cache
            result = await cache.get(full_key)
            if result is not None:
                return result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None:
                await cache.set(full_key, result, ttl_seconds)

            return result

        return wrapper

    return decorator


def invalidate_on_update(patterns: List[str]):
    """
    Decorator to invalidate cache patterns after function execution.

    Usage:
        @invalidate_on_update(["sens:{user_id}:*"])
        async def update_sensitivity(user_id: str, data: dict):
            ...
    """

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            cache = await get_cache_service()

            # Invalidate patterns (substitute variables)
            for pattern in patterns:
                # Simple variable substitution from kwargs
                resolved_pattern = pattern
                for key, value in kwargs.items():
                    resolved_pattern = resolved_pattern.replace(
                        f"{{{key}}}", str(value)
                    )
                await cache.delete_pattern(resolved_pattern)

            return result

        return wrapper

    return decorator
