import { Response, NextFunction } from 'express';
import prisma from '../config/database';
import { AdminAuthenticatedRequest } from '../types';
import { logger } from '../config/logger';

/**
 * Middleware factory for audit logging admin actions
 * Creates an audit log entry after the response is sent
 *
 * @param action - The action being performed (e.g., 'USER_LOOKUP', 'SUBSCRIPTION_GRANT')
 */
export function auditLog(action: string) {
  return (
    req: AdminAuthenticatedRequest,
    res: Response,
    next: NextFunction
  ): void => {
    // Capture start time for response timing
    const startTime = Date.now();

    // Store original JSON method to capture response
    const originalJson = res.json.bind(res);

    // Track if response was successful
    let responseStatus: number | undefined;

    // Override json to capture response status
    res.json = function (body: unknown) {
      responseStatus = res.statusCode;
      return originalJson(body);
    };

    // Log after response is sent (non-blocking)
    res.on('finish', () => {
      // Only log if we have an authenticated admin user
      if (!req.adminUser) {
        return;
      }

      // Extract target information from request
      const targetType = extractTargetType(req.path);
      const targetId = extractTargetId(req);

      // Build details object
      const details: Record<string, unknown> = {
        method: req.method,
        path: req.path,
        query: sanitizeQuery(req.query),
        responseStatus: responseStatus || res.statusCode,
        duration: Date.now() - startTime,
      };

      // Include request body for write operations (sanitized)
      if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(req.method)) {
        details.body = sanitizeBody(req.body);
      }

      // Create audit log entry asynchronously
      createAuditLogEntry({
        adminUserId: req.adminUser.id,
        action,
        targetType,
        targetId,
        details,
        ipAddress: getClientIp(req),
        userAgent: req.headers['user-agent'] || null,
      }).catch((error) => {
        // Log error but don't fail the request
        logger.error({ error, action }, 'Failed to create audit log entry');
      });
    });

    next();
  };
}

/**
 * Create audit log entry in database
 */
async function createAuditLogEntry(data: {
  adminUserId: string;
  action: string;
  targetType: string | null;
  targetId: string | null;
  details: Record<string, unknown>;
  ipAddress: string;
  userAgent: string | null;
}): Promise<void> {
  await prisma.adminAuditLog.create({
    data: {
      adminUserId: data.adminUserId,
      action: data.action,
      targetType: data.targetType,
      targetId: data.targetId,
      details: data.details as object, // Cast to satisfy Prisma Json type
      ipAddress: data.ipAddress,
      userAgent: data.userAgent,
    },
  });
}

/**
 * Extract target type from request path
 */
function extractTargetType(path: string): string | null {
  // Match common admin API patterns
  const patterns = [
    { regex: /\/api\/admin\/users/, type: 'User' },
    { regex: /\/api\/admin\/subscriptions/, type: 'Subscription' },
    { regex: /\/api\/admin\/webhooks/, type: 'WebhookEvent' },
    { regex: /\/api\/admin\/feature-flags/, type: 'FeatureFlag' },
    { regex: /\/api\/admin\/audit-logs/, type: 'AuditLog' },
  ];

  for (const pattern of patterns) {
    if (pattern.regex.test(path)) {
      return pattern.type;
    }
  }

  return null;
}

/**
 * Extract target ID from request params
 */
function extractTargetId(req: AdminAuthenticatedRequest): string | null {
  // Check for common ID parameter names
  const idParams = ['id', 'userId', 'subscriptionId', 'webhookId'];

  for (const param of idParams) {
    if (req.params[param]) {
      return req.params[param];
    }
  }

  return null;
}

/**
 * Get client IP address
 */
function getClientIp(req: AdminAuthenticatedRequest): string {
  // Check for forwarded IP (behind proxy/load balancer)
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) {
    const forwardedStr = Array.isArray(forwarded) ? forwarded[0] : forwarded;
    return forwardedStr.split(',')[0].trim();
  }

  return req.ip || req.socket.remoteAddress || 'unknown';
}

/**
 * Sanitize query parameters for logging (remove sensitive data)
 */
function sanitizeQuery(
  query: Record<string, unknown>
): Record<string, unknown> {
  const sensitiveKeys = ['password', 'token', 'secret', 'key', 'auth'];
  const sanitized: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(query)) {
    if (sensitiveKeys.some((k) => key.toLowerCase().includes(k))) {
      sanitized[key] = '[REDACTED]';
    } else {
      sanitized[key] = value;
    }
  }

  return sanitized;
}

/**
 * Sanitize request body for logging (remove sensitive data)
 */
function sanitizeBody(body: unknown): unknown {
  if (!body || typeof body !== 'object') {
    return body;
  }

  const sensitiveKeys = [
    'password',
    'passwordHash',
    'token',
    'secret',
    'mfaSecret',
    'key',
    'auth',
    'authorization',
    'creditCard',
    'ssn',
  ];

  const sanitized: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(body as Record<string, unknown>)) {
    if (sensitiveKeys.some((k) => key.toLowerCase().includes(k))) {
      sanitized[key] = '[REDACTED]';
    } else if (typeof value === 'object' && value !== null) {
      sanitized[key] = sanitizeBody(value);
    } else {
      sanitized[key] = value;
    }
  }

  return sanitized;
}

// Pre-defined audit actions for consistency
export const AuditActions = {
  // Auth
  ADMIN_LOGIN: 'ADMIN_LOGIN',
  ADMIN_LOGOUT: 'ADMIN_LOGOUT',
  ADMIN_MFA_SETUP: 'ADMIN_MFA_SETUP',
  ADMIN_MFA_VERIFY: 'ADMIN_MFA_VERIFY',

  // User Management
  USER_LIST: 'USER_LIST',
  USER_VIEW: 'USER_VIEW',
  USER_EXPORT: 'USER_EXPORT',
  USER_DELETE: 'USER_DELETE',
  USER_DISABLE: 'USER_DISABLE',

  // Subscription Management
  SUBSCRIPTION_LIST: 'SUBSCRIPTION_LIST',
  SUBSCRIPTION_VIEW: 'SUBSCRIPTION_VIEW',
  SUBSCRIPTION_GRANT: 'SUBSCRIPTION_GRANT',
  SUBSCRIPTION_EXTEND: 'SUBSCRIPTION_EXTEND',
  SUBSCRIPTION_REVOKE: 'SUBSCRIPTION_REVOKE',
  SUBSCRIPTION_LOOKUP: 'SUBSCRIPTION_LOOKUP',

  // Webhook Management
  WEBHOOK_LIST: 'WEBHOOK_LIST',
  WEBHOOK_VIEW: 'WEBHOOK_VIEW',
  WEBHOOK_SEARCH: 'WEBHOOK_SEARCH',
  WEBHOOK_RETRY: 'WEBHOOK_RETRY',

  // Feature Flags
  FEATURE_FLAG_LIST: 'FEATURE_FLAG_LIST',
  FEATURE_FLAG_CREATE: 'FEATURE_FLAG_CREATE',
  FEATURE_FLAG_UPDATE: 'FEATURE_FLAG_UPDATE',
  FEATURE_FLAG_DELETE: 'FEATURE_FLAG_DELETE',

  // Analytics
  ANALYTICS_VIEW: 'ANALYTICS_VIEW',

  // Audit Logs
  AUDIT_LOG_LIST: 'AUDIT_LOG_LIST',
} as const;

export type AuditAction = (typeof AuditActions)[keyof typeof AuditActions];
