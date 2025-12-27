import { Router } from 'express';
import {
  login,
  verifyMFACode,
  logout,
  getMe,
  setupInitialAdmin,
} from '../controllers/adminAuthController';
import { listUsers, getUser, exportUser, deleteUser } from '../controllers/adminUserController';
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
import {
  listCampaigns,
  getCampaign,
  createCampaign,
  updateCampaign,
  deleteCampaign,
  cancelCampaign,
  sendCampaign,
  getNotificationAnalytics,
  getDeviceStats,
  getUserNotifications,
  sendTestNotification,
  getNotificationTemplates,
} from '../controllers/adminNotificationController';
import {
  listTemplates as listEmailTemplates,
  getTemplate as getEmailTemplate,
  createTemplate as createEmailTemplate,
  updateTemplate as updateEmailTemplate,
  deleteTemplate as deleteEmailTemplate,
  previewTemplate,
  sendTestEmail,
  listTemplateVersions,
  getTemplateVersion,
  restoreTemplateVersion,
  listCampaigns as listEmailCampaigns,
  getCampaign as getEmailCampaign,
  createCampaign as createEmailCampaign,
  updateCampaign as updateEmailCampaign,
  deleteCampaign as deleteEmailCampaign,
  sendCampaignNow,
  cancelCampaign as cancelEmailCampaign,
  listSequences,
  getSequence,
  createSequence,
  updateSequence,
  deleteSequence,
  getEmailAnalytics,
  getSubscriberStats,
} from '../controllers/adminEmailController';
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
router.post(
  '/auth/logout',
  rateLimiters.adminApi,
  requireAnyAdmin,
  auditLog(AuditActions.ADMIN_LOGOUT),
  logout
);

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
router.get('/webhooks/stats', rateLimiters.adminApi, requireAnyAdmin, getWebhookStatistics);

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

// ============================================================================
// NOTIFICATION CAMPAIGN ROUTES
// ============================================================================

/**
 * GET /api/admin/notifications/campaigns
 * List all notification campaigns
 */
router.get('/notifications/campaigns', rateLimiters.adminApi, requireAnyAdmin, listCampaigns);

/**
 * POST /api/admin/notifications/campaigns
 * Create a new notification campaign
 */
router.post('/notifications/campaigns', rateLimiters.adminApi, requireAnyAdmin, createCampaign);

/**
 * GET /api/admin/notifications/campaigns/:id
 * Get a specific campaign with detailed stats
 */
router.get('/notifications/campaigns/:id', rateLimiters.adminApi, requireAnyAdmin, getCampaign);

/**
 * PUT /api/admin/notifications/campaigns/:id
 * Update a campaign
 */
router.put('/notifications/campaigns/:id', rateLimiters.adminApi, requireAnyAdmin, updateCampaign);

/**
 * DELETE /api/admin/notifications/campaigns/:id
 * Delete a draft campaign
 */
router.delete(
  '/notifications/campaigns/:id',
  rateLimiters.adminApi,
  requireSuperAdmin,
  deleteCampaign
);

/**
 * POST /api/admin/notifications/campaigns/:id/cancel
 * Cancel a scheduled campaign
 */
router.post(
  '/notifications/campaigns/:id/cancel',
  rateLimiters.adminApi,
  requireAnyAdmin,
  cancelCampaign
);

/**
 * POST /api/admin/notifications/campaigns/:id/send
 * Send a campaign immediately
 */
router.post(
  '/notifications/campaigns/:id/send',
  rateLimiters.adminApi,
  requireSuperAdmin,
  sendCampaign
);

// ============================================================================
// NOTIFICATION ANALYTICS ROUTES
// ============================================================================

/**
 * GET /api/admin/notifications/analytics
 * Get notification delivery analytics
 */
router.get(
  '/notifications/analytics',
  rateLimiters.adminApi,
  requireAnyAdmin,
  getNotificationAnalytics
);

/**
 * GET /api/admin/notifications/devices
 * Get device registration statistics
 */
router.get('/notifications/devices', rateLimiters.adminApi, requireAnyAdmin, getDeviceStats);

// ============================================================================
// USER NOTIFICATION MANAGEMENT ROUTES
// ============================================================================

/**
 * GET /api/admin/notifications/users/:userId
 * Get a user's notification preferences and history
 */
router.get(
  '/notifications/users/:userId',
  rateLimiters.adminApi,
  requireAnyAdmin,
  getUserNotifications
);

/**
 * POST /api/admin/notifications/users/:userId/test
 * Send a test notification to a specific user
 */
router.post(
  '/notifications/users/:userId/test',
  rateLimiters.adminApi,
  requireSuperAdmin,
  sendTestNotification
);

/**
 * GET /api/admin/notifications/templates
 * Get available notification templates
 */
router.get(
  '/notifications/templates',
  rateLimiters.adminApi,
  requireAnyAdmin,
  getNotificationTemplates
);

// ============================================================================
// EMAIL MARKETING ROUTES
// ============================================================================

// --- Email Templates ---

/**
 * GET /api/admin/email/templates
 * List all email templates
 */
router.get('/email/templates', rateLimiters.adminApi, requireAnyAdmin, listEmailTemplates);

/**
 * POST /api/admin/email/templates
 * Create a new email template
 */
router.post('/email/templates', rateLimiters.adminApi, requireAnyAdmin, createEmailTemplate);

/**
 * GET /api/admin/email/templates/:id
 * Get a specific email template
 */
router.get('/email/templates/:id', rateLimiters.adminApi, requireAnyAdmin, getEmailTemplate);

/**
 * PUT /api/admin/email/templates/:id
 * Update an email template
 */
router.put('/email/templates/:id', rateLimiters.adminApi, requireAnyAdmin, updateEmailTemplate);

/**
 * DELETE /api/admin/email/templates/:id
 * Delete an email template
 */
router.delete(
  '/email/templates/:id',
  rateLimiters.adminApi,
  requireSuperAdmin,
  deleteEmailTemplate
);

/**
 * POST /api/admin/email/templates/:id/preview
 * Preview a template with test data
 */
router.post(
  '/email/templates/:id/preview',
  rateLimiters.adminApi,
  requireAnyAdmin,
  previewTemplate
);

/**
 * POST /api/admin/email/templates/:id/test
 * Send a test email
 */
router.post('/email/templates/:id/test', rateLimiters.adminApi, requireAnyAdmin, sendTestEmail);

// --- Template Versions ---

/**
 * GET /api/admin/email/templates/:id/versions
 * List all versions of a template
 */
router.get(
  '/email/templates/:id/versions',
  rateLimiters.adminApi,
  requireAnyAdmin,
  listTemplateVersions
);

/**
 * GET /api/admin/email/templates/:id/versions/:version
 * Get a specific template version
 */
router.get(
  '/email/templates/:id/versions/:version',
  rateLimiters.adminApi,
  requireAnyAdmin,
  getTemplateVersion
);

/**
 * POST /api/admin/email/templates/:id/versions/:version/restore
 * Restore a template to a specific version
 */
router.post(
  '/email/templates/:id/versions/:version/restore',
  rateLimiters.adminApi,
  requireAnyAdmin,
  restoreTemplateVersion
);

// --- Email Campaigns ---

/**
 * GET /api/admin/email/campaigns
 * List all email campaigns
 */
router.get('/email/campaigns', rateLimiters.adminApi, requireAnyAdmin, listEmailCampaigns);

/**
 * POST /api/admin/email/campaigns
 * Create a new email campaign
 */
router.post('/email/campaigns', rateLimiters.adminApi, requireAnyAdmin, createEmailCampaign);

/**
 * GET /api/admin/email/campaigns/:id
 * Get a specific email campaign with stats
 */
router.get('/email/campaigns/:id', rateLimiters.adminApi, requireAnyAdmin, getEmailCampaign);

/**
 * PUT /api/admin/email/campaigns/:id
 * Update an email campaign
 */
router.put('/email/campaigns/:id', rateLimiters.adminApi, requireAnyAdmin, updateEmailCampaign);

/**
 * DELETE /api/admin/email/campaigns/:id
 * Delete a draft email campaign
 */
router.delete(
  '/email/campaigns/:id',
  rateLimiters.adminApi,
  requireSuperAdmin,
  deleteEmailCampaign
);

/**
 * POST /api/admin/email/campaigns/:id/send
 * Send a campaign immediately
 */
router.post('/email/campaigns/:id/send', rateLimiters.adminApi, requireSuperAdmin, sendCampaignNow);

/**
 * POST /api/admin/email/campaigns/:id/cancel
 * Cancel a scheduled or sending campaign
 */
router.post(
  '/email/campaigns/:id/cancel',
  rateLimiters.adminApi,
  requireAnyAdmin,
  cancelEmailCampaign
);

// --- Email Sequences (Drip Campaigns) ---

/**
 * GET /api/admin/email/sequences
 * List all email sequences
 */
router.get('/email/sequences', rateLimiters.adminApi, requireAnyAdmin, listSequences);

/**
 * POST /api/admin/email/sequences
 * Create a new email sequence
 */
router.post('/email/sequences', rateLimiters.adminApi, requireAnyAdmin, createSequence);

/**
 * GET /api/admin/email/sequences/:id
 * Get a specific email sequence with enrollments
 */
router.get('/email/sequences/:id', rateLimiters.adminApi, requireAnyAdmin, getSequence);

/**
 * PUT /api/admin/email/sequences/:id
 * Update an email sequence
 */
router.put('/email/sequences/:id', rateLimiters.adminApi, requireAnyAdmin, updateSequence);

/**
 * DELETE /api/admin/email/sequences/:id
 * Delete an email sequence (no active enrollments)
 */
router.delete('/email/sequences/:id', rateLimiters.adminApi, requireSuperAdmin, deleteSequence);

// --- Email Analytics ---

/**
 * GET /api/admin/email/analytics
 * Get email analytics overview
 */
router.get('/email/analytics', rateLimiters.adminApi, requireAnyAdmin, getEmailAnalytics);

/**
 * GET /api/admin/email/subscribers
 * Get subscriber statistics
 */
router.get('/email/subscribers', rateLimiters.adminApi, requireAnyAdmin, getSubscriberStats);

export default router;
