/**
 * Security Middleware
 *
 * Provides:
 * 1. HTTPS redirect for production environments
 * 2. HSTS (HTTP Strict Transport Security) headers
 * 3. Security response headers
 *
 * Note: In production, this app typically runs behind a reverse proxy (nginx, cloudflare, etc.)
 * The X-Forwarded-Proto header is used to detect the original protocol.
 */

import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';

/**
 * HSTS configuration
 * @see https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Strict_Transport_Security_Cheat_Sheet.html
 */
const HSTS_MAX_AGE = 31536000; // 1 year in seconds
const HSTS_INCLUDE_SUBDOMAINS = true;
const HSTS_PRELOAD = true;

/**
 * Middleware to redirect HTTP requests to HTTPS in production
 *
 * Only active when:
 * 1. NODE_ENV is 'production'
 * 2. ENFORCE_HTTPS environment variable is set to 'true'
 *
 * Detection methods (in order of priority):
 * 1. X-Forwarded-Proto header (from reverse proxy)
 * 2. req.secure property
 * 3. req.protocol
 *
 * Whitelisted paths that skip HTTPS redirect:
 * - /health (health checks from load balancers)
 * - /health/live (liveness probe)
 */
export const httpsRedirect = (req: Request, res: Response, next: NextFunction): void => {
  // Skip in non-production environments
  if (process.env.NODE_ENV !== 'production') {
    next();
    return;
  }

  // Skip if HTTPS enforcement is disabled
  if (process.env.ENFORCE_HTTPS !== 'true') {
    next();
    return;
  }

  // Skip health check endpoints (load balancers may use HTTP for health checks)
  const healthCheckPaths = ['/health', '/health/live'];
  if (healthCheckPaths.includes(req.path)) {
    next();
    return;
  }

  // Check if request is already HTTPS
  const isHttps =
    req.secure || // Express secure flag
    req.protocol === 'https' ||
    req.headers['x-forwarded-proto'] === 'https';

  if (!isHttps) {
    const host = req.headers.host || req.hostname;
    const redirectUrl = `https://${host}${req.originalUrl}`;

    logger.info(
      {
        action: 'https_redirect',
        from: `http://${host}${req.originalUrl}`,
        to: redirectUrl,
        correlationId: req.id,
      },
      'Redirecting HTTP to HTTPS'
    );

    // Use 301 for permanent redirect (cacheable by browsers)
    res.redirect(301, redirectUrl);
    return;
  }

  next();
};

/**
 * Middleware to set security headers including HSTS
 *
 * Headers set:
 * - Strict-Transport-Security: Enforces HTTPS for future requests
 * - X-Content-Type-Options: Prevents MIME type sniffing
 * - X-Frame-Options: Prevents clickjacking
 * - X-XSS-Protection: Legacy XSS filter (still useful for older browsers)
 * - Referrer-Policy: Controls referrer information
 * - Content-Security-Policy: Basic CSP for API responses
 *
 * Note: HSTS header is only set in production when using HTTPS
 */
export const securityHeaders = (req: Request, res: Response, next: NextFunction): void => {
  // HSTS - only set in production and when request came via HTTPS
  if (process.env.NODE_ENV === 'production') {
    const isHttps =
      req.secure || req.protocol === 'https' || req.headers['x-forwarded-proto'] === 'https';

    if (isHttps) {
      let hstsValue = `max-age=${HSTS_MAX_AGE}`;
      if (HSTS_INCLUDE_SUBDOMAINS) {
        hstsValue += '; includeSubDomains';
      }
      if (HSTS_PRELOAD) {
        hstsValue += '; preload';
      }
      res.setHeader('Strict-Transport-Security', hstsValue);
    }
  }

  // Prevent MIME type sniffing
  res.setHeader('X-Content-Type-Options', 'nosniff');

  // Prevent clickjacking
  res.setHeader('X-Frame-Options', 'DENY');

  // Legacy XSS protection (for older browsers)
  res.setHeader('X-XSS-Protection', '1; mode=block');

  // Control referrer information
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

  // Basic CSP for API (no inline scripts, restrict sources)
  // Skip CSP for Bull Board admin routes which need scripts/styles to render
  if (!req.path.startsWith('/admin/queues')) {
    // Restrictive policy suitable for API-only endpoints
    res.setHeader('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'");
  }

  next();
};

/**
 * Trust proxy configuration helper
 *
 * When running behind a reverse proxy, Express needs to trust it
 * to correctly read X-Forwarded-* headers.
 *
 * Returns the recommended trust proxy setting based on environment.
 *
 * @returns Trust proxy configuration for Express
 */
export const getTrustProxyConfig = (): boolean | string | number => {
  // In production, trust the first proxy (typical setup)
  if (process.env.NODE_ENV === 'production') {
    // Use custom setting if provided, otherwise trust first proxy
    const trustProxy = process.env.TRUST_PROXY;
    if (trustProxy === 'true') return true;
    if (trustProxy === 'false') return false;
    if (trustProxy) {
      const num = Number(trustProxy);
      // Only accept positive integers for proxy hop count
      if (!isNaN(num) && num > 0 && Number.isInteger(num)) {
        return num;
      }
    }
    // Default: trust first hop (common for cloud deployments)
    return 1;
  }

  // In development, don't trust proxies
  return false;
};
