/**
 * Redis Client Configuration
 *
 * Configures ioredis client for caching with:
 * - Automatic reconnection
 * - Health checks
 * - Graceful fallback when unavailable
 */

import Redis from 'ioredis';
import { logger } from './logger';

// Environment configuration
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const REDIS_ENABLED = process.env.REDIS_ENABLED !== 'false';

let redisClient: Redis | null = null;
let isConnected = false;

/**
 * Initialize Redis connection
 */
export function initializeRedis(): Redis | null {
  if (!REDIS_ENABLED) {
    logger.info('Redis is disabled via REDIS_ENABLED=false');
    return null;
  }

  try {
    redisClient = new Redis(REDIS_URL, {
      maxRetriesPerRequest: 3,
      retryStrategy(times) {
        if (times > 3) {
          logger.warn('Redis connection failed after 3 retries, giving up');
          return null; // Stop retrying
        }
        const delay = Math.min(times * 500, 2000);
        return delay;
      },
      enableOfflineQueue: false,
      lazyConnect: true,
    });

    redisClient.on('connect', () => {
      isConnected = true;
      logger.info('Redis connected successfully');
    });

    redisClient.on('ready', () => {
      logger.info('Redis ready to accept commands');
    });

    redisClient.on('error', (err) => {
      logger.error({ error: err.message }, 'Redis error');
      isConnected = false;
    });

    redisClient.on('close', () => {
      logger.warn('Redis connection closed');
      isConnected = false;
    });

    // Attempt to connect
    redisClient.connect().catch((err) => {
      logger.warn({ error: err.message }, 'Redis connection failed, caching disabled');
      isConnected = false;
    });

    return redisClient;
  } catch (error) {
    logger.warn(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      'Failed to initialize Redis'
    );
    return null;
  }
}

/**
 * Get the Redis client instance
 */
export function getRedisClient(): Redis | null {
  return redisClient;
}

/**
 * Check if Redis is connected and available
 */
export function isRedisAvailable(): boolean {
  return REDIS_ENABLED && isConnected && redisClient !== null;
}

/**
 * Graceful shutdown
 */
export async function closeRedis(): Promise<void> {
  if (redisClient) {
    try {
      await redisClient.quit();
      logger.info('Redis connection closed gracefully');
    } catch (error) {
      logger.error({ error }, 'Error closing Redis connection');
    }
    redisClient = null;
    isConnected = false;
  }
}

/**
 * Health check for Redis
 */
export async function redisHealthCheck(): Promise<{
  healthy: boolean;
  latency_ms?: number;
  error?: string;
}> {
  if (!REDIS_ENABLED) {
    return { healthy: true, error: 'Redis disabled' };
  }

  if (!redisClient || !isConnected) {
    return { healthy: false, error: 'Redis not connected' };
  }

  const start = Date.now();
  try {
    await redisClient.ping();
    return { healthy: true, latency_ms: Date.now() - start };
  } catch (error) {
    return {
      healthy: false,
      latency_ms: Date.now() - start,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

// Initialize on import if in non-test environment
if (process.env.NODE_ENV !== 'test') {
  initializeRedis();
}
