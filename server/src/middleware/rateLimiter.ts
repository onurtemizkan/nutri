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
 * Get rate limit entry from Redis
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
    const multi = redis.multi();

    // Increment counter and set expiry atomically
    multi.incr(redisKey);
    multi.pttl(redisKey);

    const results = await multi.exec();
    if (!results) {
      return null;
    }

    const [[countErr, count], [ttlErr, ttl]] = results as [
      [Error | null, number],
      [Error | null, number],
    ];

    if (countErr || ttlErr) {
      logger.warn({ countErr, ttlErr }, 'Redis rate limit error');
      return null;
    }

    // Set expiry if key is new (TTL will be -1)
    if (ttl === -1) {
      await redis.pexpire(redisKey, windowMs);
    }

    const now = Date.now();
    const resetAt = ttl > 0 ? now + ttl : now + windowMs;

    return {
      count: count as number,
      resetAt,
      remaining: Math.max(0, (count as number) - 1),
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
