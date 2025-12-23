import { Request, Response, NextFunction } from 'express';

/**
 * Simple in-memory rate limiter
 * For production, use Redis-backed rate limiter like express-rate-limit with Redis store
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

const rateLimitStore = new Map<string, RateLimitEntry>();

// Clean up expired entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [key, entry] of rateLimitStore.entries()) {
    if (entry.resetAt < now) {
      rateLimitStore.delete(key);
    }
  }
}, 5 * 60 * 1000);

export interface RateLimitOptions {
  windowMs: number; // Time window in milliseconds
  maxRequests: number; // Maximum requests per window
  keyGenerator?: (req: Request) => string; // Custom key generator
}

/**
 * Creates a rate limiting middleware
 */
export function createRateLimiter(options: RateLimitOptions) {
  const { windowMs, maxRequests, keyGenerator } = options;

  return (req: Request, res: Response, next: NextFunction): void => {
    const key = keyGenerator ? keyGenerator(req) : req.ip || 'unknown';
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

    // Add rate limit headers
    res.setHeader('X-RateLimit-Limit', maxRequests.toString());
    res.setHeader('X-RateLimit-Remaining', Math.max(0, maxRequests - entry.count).toString());
    res.setHeader('X-RateLimit-Reset', new Date(entry.resetAt).toISOString());

    if (entry.count > maxRequests) {
      res.status(429).json({
        error: 'Too many requests',
        retryAfter: Math.ceil((entry.resetAt - now) / 1000),
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
  }),

  // Auth endpoints: 5 requests per 15 minutes (stricter for security)
  auth: createRateLimiter({
    windowMs: 15 * 60 * 1000,
    maxRequests: 5,
  }),

  // Password reset: 3 requests per hour
  passwordReset: createRateLimiter({
    windowMs: 60 * 60 * 1000,
    maxRequests: 3,
  }),

  // Data creation: 50 requests per minute
  creation: createRateLimiter({
    windowMs: 60 * 1000,
    maxRequests: 50,
  }),

  // Admin login: 5 requests per 15 minutes per IP (stricter than regular auth)
  adminAuth: createRateLimiter({
    windowMs: 15 * 60 * 1000,
    maxRequests: 5,
  }),

  // Admin API: 100 requests per 15 minutes per admin user
  adminApi: createRateLimiter({
    windowMs: 15 * 60 * 1000,
    maxRequests: 100,
    keyGenerator: (req: Request) => {
      // Use admin user ID from request if available, otherwise IP
      const adminUser = (req as Request & { adminUser?: { id: string } }).adminUser;
      return adminUser?.id || req.ip || 'unknown';
    },
  }),
};
