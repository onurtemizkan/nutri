import { Router } from 'express';
import {
  login,
  verifyMFACode,
  logout,
  getMe,
  setupInitialAdmin,
} from '../controllers/adminAuthController';
import {
  listUsers,
  getUser,
  exportUser,
  deleteUser,
} from '../controllers/adminUserController';
import {
  listSubscriptions,
  getSubscription,
  lookupSubscription,
  grantUserSubscription,
  extendUserSubscription,
  revokeUserSubscription,
} from '../controllers/adminSubscriptionController';
import {
  getAnalyticsOverview,
  getSubscribersTimeSeries,
  getRevenueTimeSeries,
} from '../controllers/adminAnalyticsController';
import {
  listWebhooks,
  getWebhookStatistics,
  searchWebhooks,
  getWebhook,
  retryWebhook,
} from '../controllers/adminWebhookController';
import { requireAnyAdmin, requireSuperAdmin } from '../middleware/adminAuth';
import { auditLog, AuditActions } from '../middleware/adminAudit';
import { rateLimiters } from '../middleware/rateLimiter';
import { ipAllowlist } from '../middleware/ipAllowlist';

const router = Router();

// ============================================================================
// SECURITY MIDDLEWARE (Applied to all admin routes)
// ============================================================================

// Apply IP allowlist to all admin routes (if configured)
router.use(ipAllowlist);

// ============================================================================
// ADMIN AUTHENTICATION ROUTES
// ============================================================================

/**
 * POST /api/admin/auth/login
 * Admin login (returns token or MFA challenge)
 */
router.post('/auth/login', rateLimiters.adminAuth, login);

/**
 * POST /api/admin/auth/mfa/verify
 * Verify MFA code and complete login
 */
router.post('/auth/mfa/verify', rateLimiters.adminAuth, verifyMFACode);

/**
 * POST /api/admin/auth/logout
 * Admin logout (client-side token invalidation)
 */
router.post('/auth/logout', rateLimiters.adminApi, requireAnyAdmin, auditLog(AuditActions.ADMIN_LOGOUT), logout);

/**
 * GET /api/admin/auth/me
 * Get current admin user info (requires auth)
 */
router.get('/auth/me', rateLimiters.adminApi, requireAnyAdmin, getMe);

/**
 * POST /api/admin/auth/setup
 * Create initial admin user (only works if no admin users exist)
 * This is for initial setup only - remove or disable in production after setup
 */
router.post('/auth/setup', setupInitialAdmin);

// ============================================================================
// USER MANAGEMENT ROUTES
// ============================================================================

/**
 * GET /api/admin/users
 * List users with pagination, search, and filters
 */
router.get(
  '/users',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.USER_LIST),
  listUsers
);

/**
 * GET /api/admin/users/:id
 * Get detailed user information
 */
router.get(
  '/users/:id',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.USER_VIEW),
  getUser
);

/**
 * POST /api/admin/users/:id/export
 * Export all user data for GDPR compliance
 */
router.post(
  '/users/:id/export',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.USER_EXPORT),
  exportUser
);

/**
 * DELETE /api/admin/users/:id
 * Delete user account for GDPR compliance
 * Requires SUPER_ADMIN role
 *
 * Body validation: { reason: string (min 10 chars) } - validated in controller
 * using deleteUserSchema from validation/adminSchemas.ts
 */
router.delete(
  '/users/:id',
  rateLimiters.adminApi,
  requireSuperAdmin,
  auditLog(AuditActions.USER_DELETE),
  deleteUser
);

// ============================================================================
// SUBSCRIPTION MANAGEMENT ROUTES
// ============================================================================

/**
 * GET /api/admin/subscriptions
 * List users with subscriptions
 */
router.get(
  '/subscriptions',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.SUBSCRIPTION_LIST),
  listSubscriptions
);

/**
 * GET /api/admin/subscriptions/lookup
 * Lookup subscription by Apple transaction ID
 * Note: This route must come before :id route to avoid conflicts
 */
router.get(
  '/subscriptions/lookup',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.SUBSCRIPTION_LOOKUP),
  lookupSubscription
);

/**
 * GET /api/admin/subscriptions/:id
 * Get detailed subscription information for a user
 */
router.get(
  '/subscriptions/:id',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.SUBSCRIPTION_VIEW),
  getSubscription
);

/**
 * POST /api/admin/subscriptions/:id/grant
 * Manually grant Pro subscription to a user
 * Requires SUPER_ADMIN role
 */
router.post(
  '/subscriptions/:id/grant',
  rateLimiters.adminApi,
  requireSuperAdmin,
  auditLog(AuditActions.SUBSCRIPTION_GRANT),
  grantUserSubscription
);

/**
 * POST /api/admin/subscriptions/:id/extend
 * Extend a user's subscription
 * Requires SUPER_ADMIN role
 */
router.post(
  '/subscriptions/:id/extend',
  rateLimiters.adminApi,
  requireSuperAdmin,
  auditLog(AuditActions.SUBSCRIPTION_EXTEND),
  extendUserSubscription
);

/**
 * POST /api/admin/subscriptions/:id/revoke
 * Revoke a user's subscription
 * Requires SUPER_ADMIN role
 */
router.post(
  '/subscriptions/:id/revoke',
  rateLimiters.adminApi,
  requireSuperAdmin,
  auditLog(AuditActions.SUBSCRIPTION_REVOKE),
  revokeUserSubscription
);

// ============================================================================
// WEBHOOK EVENT ROUTES
// ============================================================================

/**
 * GET /api/admin/webhooks
 * List webhook events with filters and pagination
 */
router.get(
  '/webhooks',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.WEBHOOK_LIST),
  listWebhooks
);

/**
 * GET /api/admin/webhooks/stats
 * Get webhook event statistics
 */
router.get(
  '/webhooks/stats',
  rateLimiters.adminApi,
  requireAnyAdmin,
  getWebhookStatistics
);

/**
 * GET /api/admin/webhooks/search
 * Search webhooks by transaction ID
 * Note: This route must come before :id to avoid conflicts
 */
router.get(
  '/webhooks/search',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.WEBHOOK_SEARCH),
  searchWebhooks
);

/**
 * GET /api/admin/webhooks/:id
 * Get detailed webhook event information
 */
router.get(
  '/webhooks/:id',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.WEBHOOK_VIEW),
  getWebhook
);

/**
 * POST /api/admin/webhooks/:id/retry
 * Retry processing a failed webhook event
 * Requires SUPER_ADMIN role
 */
router.post(
  '/webhooks/:id/retry',
  rateLimiters.adminApi,
  requireSuperAdmin,
  auditLog(AuditActions.WEBHOOK_RETRY),
  retryWebhook
);

// ============================================================================
// ANALYTICS ROUTES
// ============================================================================

/**
 * GET /api/admin/analytics/overview
 * Get comprehensive subscription analytics
 */
router.get(
  '/analytics/overview',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.ANALYTICS_VIEW),
  getAnalyticsOverview
);

/**
 * GET /api/admin/analytics/subscribers-over-time
 * Get daily subscriber counts
 */
router.get(
  '/analytics/subscribers-over-time',
  rateLimiters.adminApi,
  requireAnyAdmin,
  getSubscribersTimeSeries
);

/**
 * GET /api/admin/analytics/revenue-over-time
 * Get monthly revenue data
 */
router.get(
  '/analytics/revenue-over-time',
  rateLimiters.adminApi,
  requireAnyAdmin,
  getRevenueTimeSeries
);

// ============================================================================
// AUDIT LOG ROUTES
// ============================================================================

// GET /api/admin/audit-logs - List audit logs with filters

export default router;
