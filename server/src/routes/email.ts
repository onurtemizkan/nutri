/**
 * Email Routes - Webhook handlers and user preference endpoints
 */

import { Router, Request, Response } from 'express';
import { PrismaClient, EmailFrequency } from '@prisma/client';
import crypto from 'crypto';
import { z } from 'zod';
import { authenticate } from '../middleware/auth';
import { AuthenticatedRequest } from '../types';
import {
  handleEmailWebhook,
  unsubscribeUser,
  ensureEmailPreferences,
} from '../services/emailService';
import { triggerSequenceEvent } from '../services/emailQueueService';
import { logger } from '../config/logger';
import { HTTP_STATUS, ERROR_MESSAGES } from '../config/constants';

const router = Router();
const prisma = new PrismaClient();

// Resend webhook secret for verification
const RESEND_WEBHOOK_SECRET = process.env.RESEND_WEBHOOK_SECRET;

// =============================================================================
// Webhook Endpoints (No Auth - Verified by Signature)
// =============================================================================

/**
 * POST /api/email/webhooks/resend
 * Handle Resend webhook events (delivery, open, click, bounce, complaint)
 */
router.post('/webhooks/resend', async (req: Request, res: Response) => {
  try {
    // Verify webhook signature
    if (RESEND_WEBHOOK_SECRET) {
      const signature = req.headers['resend-signature'] as string;
      const timestamp = req.headers['resend-timestamp'] as string;

      if (!signature || !timestamp) {
        logger.warn('Missing webhook signature headers');
        return res.status(HTTP_STATUS.UNAUTHORIZED).json({ error: 'Missing signature' });
      }

      // Verify signature (HMAC-SHA256)
      const payload = `${timestamp}.${JSON.stringify(req.body)}`;
      const expectedSignature = crypto
        .createHmac('sha256', RESEND_WEBHOOK_SECRET)
        .update(payload)
        .digest('hex');

      if (signature !== expectedSignature) {
        logger.warn('Invalid webhook signature');
        return res.status(HTTP_STATUS.UNAUTHORIZED).json({ error: 'Invalid signature' });
      }

      // Check timestamp to prevent replay attacks (5 minute window)
      const timestampDate = new Date(timestamp);
      const now = new Date();
      const diff = now.getTime() - timestampDate.getTime();
      if (diff > 5 * 60 * 1000) {
        logger.warn('Webhook timestamp too old');
        return res.status(HTTP_STATUS.UNAUTHORIZED).json({ error: 'Timestamp expired' });
      }
    }

    const { type, data } = req.body;

    logger.info({ type, emailId: data?.email_id }, 'Received Resend webhook');

    // Process webhook event
    await handleEmailWebhook(type, data);

    return res.status(HTTP_STATUS.OK).json({ received: true });
  } catch (error) {
    logger.error({ error }, 'Webhook processing error');
    // Return 200 to prevent retries for processing errors
    return res.status(HTTP_STATUS.OK).json({ received: true, error: 'Processing error' });
  }
});

/**
 * GET /api/email/unsubscribe/:token
 * One-click unsubscribe endpoint
 */
router.get('/unsubscribe/:token', async (req: Request, res: Response) => {
  try {
    const { token } = req.params;

    const result = await unsubscribeUser(token);

    if (!result.success) {
      return res.status(HTTP_STATUS.BAD_REQUEST).json({ error: result.error });
    }

    // Return a simple HTML page
    return res.send(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>Unsubscribed - Nutri</title>
          <style>
            body { font-family: system-ui, sans-serif; max-width: 500px; margin: 100px auto; text-align: center; }
            h1 { color: #10B981; }
            p { color: #6B7280; }
          </style>
        </head>
        <body>
          <h1>You've been unsubscribed</h1>
          <p>You will no longer receive marketing emails from Nutri.</p>
          <p>If this was a mistake, you can update your preferences in the Nutri app.</p>
        </body>
      </html>
    `);
  } catch (error) {
    logger.error({ error }, 'Unsubscribe error');
    return res
      .status(HTTP_STATUS.INTERNAL_SERVER_ERROR)
      .json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
  }
});

/**
 * POST /api/email/unsubscribe/:token
 * One-click unsubscribe (List-Unsubscribe-Post)
 */
router.post('/unsubscribe/:token', async (req: Request, res: Response) => {
  try {
    const { token } = req.params;

    const result = await unsubscribeUser(token);

    if (!result.success) {
      return res.status(HTTP_STATUS.BAD_REQUEST).json({ error: result.error });
    }

    return res.status(HTTP_STATUS.OK).json({ success: true });
  } catch (error) {
    logger.error({ error }, 'Unsubscribe error');
    return res
      .status(HTTP_STATUS.INTERNAL_SERVER_ERROR)
      .json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
  }
});

// =============================================================================
// User Preference Endpoints (Authenticated)
// =============================================================================

// Validation schemas
const updatePreferencesSchema = z.object({
  categories: z
    .object({
      weekly_reports: z.boolean().optional(),
      health_insights: z.boolean().optional(),
      tips: z.boolean().optional(),
      features: z.boolean().optional(),
      promotions: z.boolean().optional(),
    })
    .optional(),
  frequency: z.nativeEnum(EmailFrequency).optional(),
  marketingOptIn: z.boolean().optional(),
});

/**
 * GET /api/email/preferences
 * Get user's email preferences
 */
router.get('/preferences', authenticate, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const userId = req.userId!;

    // Ensure preferences exist
    await ensureEmailPreferences(userId);

    const preferences = await prisma.emailPreference.findUnique({
      where: { userId },
      select: {
        categories: true,
        frequency: true,
        marketingOptIn: true,
        doubleOptInConfirmedAt: true,
        globalUnsubscribedAt: true,
        isSuppressed: true,
        engagementScore: true,
        lastEngagementAt: true,
        createdAt: true,
        updatedAt: true,
      },
    });

    return res.status(HTTP_STATUS.OK).json(preferences);
  } catch (error) {
    logger.error({ error }, 'Get preferences error');
    return res
      .status(HTTP_STATUS.INTERNAL_SERVER_ERROR)
      .json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
  }
});

/**
 * PUT /api/email/preferences
 * Update user's email preferences
 */
router.put('/preferences', authenticate, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const userId = req.userId!;

    const validation = updatePreferencesSchema.safeParse(req.body);
    if (!validation.success) {
      return res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid request',
        details: validation.error.errors,
      });
    }

    const { categories, frequency, marketingOptIn } = validation.data;

    // Ensure preferences exist
    await ensureEmailPreferences(userId);

    // Get current preferences
    const current = await prisma.emailPreference.findUnique({
      where: { userId },
    });

    const updateData: Record<string, unknown> = {};

    if (categories !== undefined) {
      // Merge with existing categories
      const existingCategories = (current?.categories as Record<string, boolean>) || {};
      updateData.categories = { ...existingCategories, ...categories };
    }

    if (frequency !== undefined) {
      updateData.frequency = frequency;
    }

    if (marketingOptIn !== undefined) {
      updateData.marketingOptIn = marketingOptIn;

      // If opting in, clear unsubscribe timestamp
      if (marketingOptIn) {
        updateData.globalUnsubscribedAt = null;
      }
    }

    const updated = await prisma.emailPreference.update({
      where: { userId },
      data: updateData,
    });

    logger.info({ userId }, 'Email preferences updated');

    return res.status(HTTP_STATUS.OK).json({
      categories: updated.categories,
      frequency: updated.frequency,
      marketingOptIn: updated.marketingOptIn,
      doubleOptInConfirmedAt: updated.doubleOptInConfirmedAt,
      globalUnsubscribedAt: updated.globalUnsubscribedAt,
    });
  } catch (error) {
    logger.error({ error }, 'Update preferences error');
    return res
      .status(HTTP_STATUS.INTERNAL_SERVER_ERROR)
      .json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
  }
});

/**
 * POST /api/email/opt-in
 * Confirm marketing opt-in (double opt-in confirmation)
 */
router.post('/opt-in', authenticate, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const userId = req.userId!;

    await ensureEmailPreferences(userId);

    await prisma.emailPreference.update({
      where: { userId },
      data: {
        marketingOptIn: true,
        doubleOptInConfirmedAt: new Date(),
        globalUnsubscribedAt: null,
      },
    });

    // Trigger welcome sequence for marketing opt-in
    await triggerSequenceEvent(userId, 'SIGNUP');

    logger.info({ userId }, 'User opted in to marketing');

    return res.status(HTTP_STATUS.OK).json({ success: true });
  } catch (error) {
    logger.error({ error }, 'Opt-in error');
    return res
      .status(HTTP_STATUS.INTERNAL_SERVER_ERROR)
      .json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
  }
});

/**
 * POST /api/email/opt-out
 * Opt out of all marketing emails
 */
router.post('/opt-out', authenticate, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const userId = req.userId!;

    await ensureEmailPreferences(userId);

    await prisma.emailPreference.update({
      where: { userId },
      data: {
        marketingOptIn: false,
        globalUnsubscribedAt: new Date(),
      },
    });

    // Exit all active sequences
    await prisma.emailSequenceEnrollment.updateMany({
      where: {
        userId,
        status: 'ACTIVE',
      },
      data: {
        status: 'EXITED',
        exitedAt: new Date(),
        exitReason: 'user_unsubscribed',
      },
    });

    logger.info({ userId }, 'User opted out of marketing');

    return res.status(HTTP_STATUS.OK).json({ success: true });
  } catch (error) {
    logger.error({ error }, 'Opt-out error');
    return res
      .status(HTTP_STATUS.INTERNAL_SERVER_ERROR)
      .json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
  }
});

/**
 * GET /api/email/history
 * Get user's email history
 */
router.get('/history', authenticate, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const userId = req.userId!;
    const limit = Math.min(parseInt(req.query.limit as string) || 20, 50);
    const offset = parseInt(req.query.offset as string) || 0;

    const [logs, total] = await Promise.all([
      prisma.emailLog.findMany({
        where: { userId },
        orderBy: { createdAt: 'desc' },
        take: limit,
        skip: offset,
        select: {
          id: true,
          templateSlug: true,
          status: true,
          sentAt: true,
          deliveredAt: true,
          openedAt: true,
          clickedAt: true,
          createdAt: true,
        },
      }),
      prisma.emailLog.count({ where: { userId } }),
    ]);

    return res.status(HTTP_STATUS.OK).json({
      logs,
      total,
      limit,
      offset,
      hasMore: offset + logs.length < total,
    });
  } catch (error) {
    logger.error({ error }, 'Get email history error');
    return res
      .status(HTTP_STATUS.INTERNAL_SERVER_ERROR)
      .json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
  }
});

export default router;
