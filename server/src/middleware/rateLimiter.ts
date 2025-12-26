import { Request, Response, NextFunction } from 'express';
import { getRedisClient, isRedisAvailable } from '../config/redis';
import { logger } from '../config/logger';

/**
 * Distributed rate limiter with Redis backend and in-memory fallback
 *
 * Uses Redis for consistent rate limiting across multiple server instances.
 * Falls back to in-memory store when Redis is unavailable.
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

// In-memory fallback store
const rateLimitStore = new Map<string, RateLimitEntry>();

// Clean up expired entries every 5 minutes
setInterval(
  () => {
    const now = Date.now();
    for (const [key, entry] of rateLimitStore.entries()) {
      if (entry.resetAt < now) {
        rateLimitStore.delete(key);
      }
    }
  },
  5 * 60 * 1000
);

export interface RateLimitOptions {
  windowMs: number; // Time window in milliseconds
  maxRequests: number; // Maximum requests per window
  keyGenerator?: (req: Request) => string; // Custom key generator
  keyPrefix?: string; // Redis key prefix (default: 'ratelimit:')
}

/**
 * Lua script for atomic rate limiting
 *
 * This script atomically:
 * 1. Increments the counter
 * 2. Sets expiry if the key is new (TTL = -1)
 * 3. Returns both the count and remaining TTL
 *
 * This prevents the race condition where INCR succeeds but PEXPIRE fails,
 * which would leave the key without a TTL (persisting forever).
 */
const RATE_LIMIT_LUA_SCRIPT = `
  local key = KEYS[1]
  local window_ms = tonumber(ARGV[1])
  local count = redis.call('INCR', key)
  local ttl = redis.call('PTTL', key)
  if ttl == -1 then
    redis.call('PEXPIRE', key, window_ms)
    ttl = window_ms
  end
  return {count, ttl}
`;

/**
 * Get rate limit entry from Redis using atomic Lua script
 */
async function getRedisEntry(
  key: string,
  windowMs: number
): Promise<{ count: number; resetAt: number; remaining: number } | null> {
  const redis = getRedisClient();
  if (!redis || !isRedisAvailable()) {
    return null;
  }

  try {
    const redisKey = `ratelimit:${key}`;

    // Execute Lua script atomically - INCR and conditional PEXPIRE happen together
    const result = (await redis.eval(RATE_LIMIT_LUA_SCRIPT, 1, redisKey, windowMs.toString())) as [
      number,
      number,
    ];

    if (!result || !Array.isArray(result) || result.length !== 2) {
      logger.warn({ result }, 'Unexpected Redis rate limit script result');
      return null;
    }

    const [count, ttl] = result;
    const now = Date.now();
    const resetAt = now + ttl;

    return {
      count,
      resetAt,
      remaining: Math.max(0, count - 1),
    };
  } catch (error) {
    logger.warn({ error }, 'Redis rate limit operation failed, using in-memory fallback');
    return null;
  }
}

/**
 * Get rate limit entry from in-memory store
 */
function getMemoryEntry(key: string, windowMs: number): RateLimitEntry {
  const now = Date.now();
  let entry = rateLimitStore.get(key);

  // Create new entry or reset if window expired
  if (!entry || entry.resetAt < now) {
    entry = {
      count: 0,
      resetAt: now + windowMs,
    };
    rateLimitStore.set(key, entry);
  }

  entry.count++;
  return entry;
}

/**
 * Creates a rate limiting middleware with Redis support
 *
 * Uses Redis for distributed rate limiting when available.
 * Falls back to in-memory when Redis is unavailable.
 */
export function createRateLimiter(options: RateLimitOptions) {
  const { windowMs, maxRequests, keyGenerator, keyPrefix = 'default' } = options;

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    const baseKey = keyGenerator ? keyGenerator(req) : req.ip || 'unknown';
    const key = `${keyPrefix}:${baseKey}`;

    let count: number;
    let resetAt: number;

    // Try Redis first, fall back to memory
    const redisEntry = await getRedisEntry(key, windowMs);

    if (redisEntry) {
      count = redisEntry.count;
      resetAt = redisEntry.resetAt;
    } else {
      const memEntry = getMemoryEntry(key, windowMs);
      count = memEntry.count;
      resetAt = memEntry.resetAt;
    }

    // Add rate limit headers
    res.setHeader('X-RateLimit-Limit', maxRequests.toString());
    res.setHeader('X-RateLimit-Remaining', Math.max(0, maxRequests - count).toString());
    res.setHeader('X-RateLimit-Reset', new Date(resetAt).toISOString());

    if (count > maxRequests) {
      const now = Date.now();
      res.status(429).json({
        error: 'Too many requests',
        retryAfter: Math.ceil((resetAt - now) / 1000),
      });
      return;
    }

    next();
  };
}

/**
 * Preset rate limiters for common use cases
 */
export const rateLimiters = {
  // General API rate limit: 100 requests per 15 minutes
  api: createRateLimiter({
    windowMs: 15 * 60 * 1000,
    maxRequests: 100,
    keyPrefix: 'api',
  }),

  // Auth endpoints: 5 requests per 15 minutes (stricter for security)
  auth: createRateLimiter({
    windowMs: 15 * 60 * 1000,
    maxRequests: 5,
    keyPrefix: 'auth',
  }),

  // Password reset: 3 requests per hour
  passwordReset: createRateLimiter({
    windowMs: 60 * 60 * 1000,
    maxRequests: 3,
    keyPrefix: 'pwreset',
  }),

  // Data creation: 50 requests per minute
  creation: createRateLimiter({
    windowMs: 60 * 1000,
    maxRequests: 50,
    keyPrefix: 'create',
  }),

  // Admin login: 15 requests per 1 minute per IP
  adminAuth: createRateLimiter({
    windowMs: 60 * 1000,
    maxRequests: 15,
    keyPrefix: 'admin-auth',
  }),

  // Admin API: 100 requests per 15 minutes per admin user
  adminApi: createRateLimiter({
    windowMs: 15 * 60 * 1000,
    maxRequests: 100,
    keyPrefix: 'admin-api',
    keyGenerator: (req: Request) => {
      // Use admin user ID from request if available, otherwise IP
      const adminUser = (req as Request & { adminUser?: { id: string } }).adminUser;
      return adminUser?.id || req.ip || 'unknown';
    },
  }),
};
