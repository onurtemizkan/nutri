/**
 * Email Webhook Routes
 *
 * Handles webhook endpoints for email events from Resend.
 */

import { Router } from 'express';
import { handleResendWebhook, getWebhookStats } from '../controllers/emailWebhookController';
import { authenticate } from '../middleware/auth';
import { createRateLimiter } from '../middleware/rateLimiter';

const router = Router();

// Rate limiter for webhook endpoint
const webhookRateLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 1000, // 1000 requests per minute
});

/**
 * POST /api/webhooks/email/resend
 * Resend webhook endpoint
 * No authentication (uses signature verification)
 */
router.post('/resend', webhookRateLimiter, handleResendWebhook);

/**
 * GET /api/webhooks/email/stats
 * Get email webhook statistics (admin only - requires authentication)
 * Note: In production, add admin role check middleware
 */
router.get('/stats', authenticate, getWebhookStats);

export default router;
