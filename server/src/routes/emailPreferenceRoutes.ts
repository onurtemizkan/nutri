/**
 * Email Preference Routes
 *
 * Handles user email preference endpoints including:
 * - Get/update preferences (authenticated)
 * - Unsubscribe (token-based, no auth required)
 * - Double opt-in confirmation (token-based)
 */

import { Router } from 'express';
import {
  getPreferences,
  updatePreferences,
  confirmDoubleOptIn,
  unsubscribePage,
  processUnsubscribe,
  oneClickUnsubscribe,
  requestDoubleOptIn,
  resubscribe,
} from '../controllers/emailPreferenceController';
import { authenticate } from '../middleware/auth';
import { createRateLimiter } from '../middleware/rateLimiter';

const router = Router();

// Rate limiters
const preferencesRateLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 30, // 30 requests per 15 minutes
});

const unsubscribeRateLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 10, // 10 requests per minute
});

// =============================================================================
// Authenticated Routes (require JWT)
// =============================================================================

/**
 * GET /api/email/preferences
 * Get current user's email preferences
 */
router.get('/preferences', authenticate, preferencesRateLimiter, getPreferences);

/**
 * PUT /api/email/preferences
 * Update email preferences
 */
router.put('/preferences', authenticate, preferencesRateLimiter, updatePreferences);

/**
 * POST /api/email/opt-in
 * Request double opt-in confirmation email
 */
router.post('/opt-in', authenticate, preferencesRateLimiter, requestDoubleOptIn);

/**
 * POST /api/email/resubscribe
 * Resubscribe to marketing emails (for users who previously unsubscribed)
 */
router.post('/resubscribe', authenticate, preferencesRateLimiter, resubscribe);

// =============================================================================
// Public Routes (token-based authentication)
// =============================================================================

/**
 * GET /api/email/opt-in/confirm
 * Confirm double opt-in (from email link)
 */
router.get('/opt-in/confirm', unsubscribeRateLimiter, confirmDoubleOptIn);

/**
 * GET /api/email/unsubscribe/:token
 * Show unsubscribe preferences page (from email link)
 */
router.get('/unsubscribe/:token', unsubscribeRateLimiter, unsubscribePage);

/**
 * POST /api/email/unsubscribe
 * Process unsubscribe form submission
 */
router.post('/unsubscribe', unsubscribeRateLimiter, processUnsubscribe);

/**
 * POST /api/email/unsubscribe/one-click
 * One-click unsubscribe (RFC 8058)
 * Used by email clients that support List-Unsubscribe-Post header
 */
router.post('/unsubscribe/one-click', unsubscribeRateLimiter, oneClickUnsubscribe);

export default router;
