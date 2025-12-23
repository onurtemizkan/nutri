import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';
import prisma from '../config/database';

/**
 * IP Allowlist Middleware for Admin Endpoints
 *
 * Optional security feature that restricts admin access to specific IP addresses.
 * Configure via ADMIN_IP_ALLOWLIST environment variable (comma-separated IPs).
 *
 * If ADMIN_IP_ALLOWLIST is not set, all IPs are allowed.
 *
 * Example: ADMIN_IP_ALLOWLIST=192.168.1.1,10.0.0.1,::1,127.0.0.1
 */

const ADMIN_IP_ALLOWLIST = process.env.ADMIN_IP_ALLOWLIST;

// Parse allowlist on startup
const allowedIPs: Set<string> = new Set();
if (ADMIN_IP_ALLOWLIST) {
  ADMIN_IP_ALLOWLIST.split(',')
    .map((ip) => ip.trim())
    .filter((ip) => ip.length > 0)
    .forEach((ip) => allowedIPs.add(ip));

  logger.info({ allowedIPs: Array.from(allowedIPs) }, 'IP allowlist configured');
}

/**
 * Normalize IP address for comparison
 * Handles IPv4-mapped IPv6 addresses (::ffff:127.0.0.1 -> 127.0.0.1)
 */
function normalizeIP(ip: string): string {
  if (ip.startsWith('::ffff:')) {
    return ip.substring(7);
  }
  return ip;
}

/**
 * Log blocked IP access attempt to audit log
 */
async function logBlockedAccess(
  ip: string,
  path: string,
  userAgent: string | undefined
): Promise<void> {
  try {
    await prisma.adminAuditLog.create({
      data: {
        adminUserId: 'system', // No user for blocked requests
        action: 'IP_BLOCKED',
        targetType: 'AdminAccess',
        targetId: null,
        details: {
          blockedIP: ip,
          requestPath: path,
          allowedIPs: Array.from(allowedIPs),
        },
        ipAddress: ip,
        userAgent: userAgent || null,
      },
    });
  } catch (error) {
    logger.error({ error, ip }, 'Failed to log blocked IP access');
  }
}

/**
 * IP Allowlist middleware
 *
 * Returns 403 Forbidden if IP is not in allowlist.
 * Logs blocked attempts to AdminAuditLog.
 */
export function ipAllowlist(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // If no allowlist configured, allow all
  if (allowedIPs.size === 0) {
    next();
    return;
  }

  const clientIP = normalizeIP(req.ip || 'unknown');

  // Check if IP is allowed
  if (allowedIPs.has(clientIP) || allowedIPs.has(req.ip || '')) {
    next();
    return;
  }

  // Log blocked access asynchronously (don't delay response)
  logBlockedAccess(clientIP, req.path, req.get('user-agent'));

  logger.warn({ clientIP, requestPath: req.path }, 'Admin access blocked by IP allowlist');

  res.status(403).json({
    error: 'Access denied',
    message: 'Your IP address is not authorized to access admin resources',
  });
}

/**
 * Check if IP allowlist is enabled
 */
export function isIPAllowlistEnabled(): boolean {
  return allowedIPs.size > 0;
}

/**
 * Get the list of allowed IPs (for debugging/admin UI)
 */
export function getAllowedIPs(): string[] {
  return Array.from(allowedIPs);
}
