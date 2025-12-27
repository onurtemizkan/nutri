/**
 * Email Preference Controller
 *
 * Handles user email preferences including:
 * - Get/update preferences
 * - Category management
 * - Unsubscribe (one-click and web form)
 * - Double opt-in confirmation
 */

import { Request, Response } from 'express';
import { z } from 'zod';
import prisma from '../config/database';
import { logger } from '../config/logger';
import { verifyUnsubscribeToken, generateUnsubscribeToken } from '../utils/emailHelpers';
import { sendTransactional, enrollInSequence } from '../services/emailService';
import { MARKETING_CATEGORIES, EMAIL_CONFIG } from '../config/resend';
import { AuthenticatedRequest } from '../types';

/**
 * Schema for updating email preferences
 */
const updatePreferencesSchema = z.object({
  categories: z
    .object({
      weekly_reports: z.boolean().optional(),
      health_insights: z.boolean().optional(),
      tips: z.boolean().optional(),
      features: z.boolean().optional(),
      promotions: z.boolean().optional(),
      newsletter: z.boolean().optional(),
    })
    .optional(),
  frequency: z.enum(['REALTIME', 'DAILY_DIGEST', 'WEEKLY_DIGEST']).optional(),
  marketingOptIn: z.boolean().optional(),
});

/**
 * Get current user's email preferences
 */
export async function getPreferences(req: AuthenticatedRequest, res: Response): Promise<void> {
  const userId = req.userId;

  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  // Get or create preferences
  let preference = await prisma.emailPreference.findUnique({
    where: { userId },
  });

  if (!preference) {
    // Create default preferences
    preference = await prisma.emailPreference.create({
      data: {
        userId,
        categories: {
          weekly_reports: true,
          health_insights: true,
          tips: true,
          features: true,
          promotions: false,
          newsletter: false,
        },
        frequency: 'REALTIME',
        marketingOptIn: false,
      },
    });
  }

  res.json({
    categories: preference.categories,
    frequency: preference.frequency,
    marketingOptIn: preference.marketingOptIn,
    doubleOptInConfirmed: !!preference.doubleOptInConfirmedAt,
    globalUnsubscribed: !!preference.globalUnsubscribedAt,
    availableCategories: MARKETING_CATEGORIES,
  });
}

/**
 * Update email preferences
 */
export async function updatePreferences(req: AuthenticatedRequest, res: Response): Promise<void> {
  const userId = req.userId;

  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  // Validate input
  const result = updatePreferencesSchema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json({ error: 'Invalid input', details: result.error.format() });
    return;
  }

  const { categories, frequency, marketingOptIn } = result.data;

  // Build update data
  const updateData: Record<string, unknown> = {};

  if (categories) {
    // Merge with existing categories
    const existing = await prisma.emailPreference.findUnique({
      where: { userId },
      select: { categories: true },
    });
    const existingCategories = (existing?.categories as Record<string, boolean>) || {};
    updateData.categories = { ...existingCategories, ...categories };
  }

  if (frequency) {
    updateData.frequency = frequency;
  }

  if (marketingOptIn !== undefined) {
    updateData.marketingOptIn = marketingOptIn;

    // If opting in, send double opt-in confirmation
    if (marketingOptIn && !req.body.skipDoubleOptIn) {
      // Will trigger double opt-in email
      updateData.doubleOptInConfirmedAt = null;
    }
  }

  // Upsert preferences
  const preference = await prisma.emailPreference.upsert({
    where: { userId },
    update: updateData,
    create: {
      userId,
      ...updateData,
      categories: updateData.categories || {
        weekly_reports: true,
        health_insights: true,
        tips: true,
        features: true,
        promotions: false,
        newsletter: false,
      },
    },
  });

  // If marketing opt-in was just enabled, send double opt-in email
  if (marketingOptIn && !preference.doubleOptInConfirmedAt) {
    try {
      await sendTransactional(userId, 'double-opt-in-confirmation', {});
      logger.info({ userId }, 'Double opt-in confirmation email sent');
    } catch (error) {
      logger.error({ userId, error }, 'Failed to send double opt-in email');
    }
  }

  res.json({
    message: 'Preferences updated',
    categories: preference.categories,
    frequency: preference.frequency,
    marketingOptIn: preference.marketingOptIn,
    doubleOptInConfirmed: !!preference.doubleOptInConfirmedAt,
  });
}

/**
 * Confirm double opt-in
 * GET /api/email/opt-in/confirm?token=xxx
 */
export async function confirmDoubleOptIn(req: Request, res: Response): Promise<void> {
  const { token } = req.query;

  if (typeof token !== 'string') {
    res.status(400).json({ error: 'Invalid token' });
    return;
  }

  const decoded = verifyUnsubscribeToken(token);
  if (!decoded) {
    res.status(400).json({ error: 'Invalid or expired token' });
    return;
  }

  const { userId } = decoded;

  // Update preferences
  await prisma.emailPreference.update({
    where: { userId },
    data: {
      doubleOptInConfirmedAt: new Date(),
      marketingOptIn: true,
    },
  });

  // Enroll in onboarding sequence if not already
  await enrollInSequence(userId, 'SIGNUP');

  logger.info({ userId }, 'Double opt-in confirmed');

  // Return success page HTML
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Email Confirmed | Nutri</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 600px; margin: 50px auto; padding: 20px; text-align: center; }
        .success { color: #10b981; font-size: 48px; margin-bottom: 20px; }
        h1 { color: #1f2937; margin-bottom: 10px; }
        p { color: #6b7280; }
      </style>
    </head>
    <body>
      <div class="success">✓</div>
      <h1>You're all set!</h1>
      <p>Your email preferences have been confirmed. You'll now receive updates from Nutri.</p>
    </body>
    </html>
  `);
}

/**
 * Unsubscribe page
 * GET /api/email/unsubscribe/:token
 */
export async function unsubscribePage(req: Request, res: Response): Promise<void> {
  const { token } = req.params;

  const decoded = verifyUnsubscribeToken(token);
  if (!decoded) {
    res.status(400).send(`
      <!DOCTYPE html>
      <html>
      <head><title>Invalid Link | Nutri</title></head>
      <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto; text-align: center;">
        <h1>Invalid or Expired Link</h1>
        <p>This unsubscribe link is no longer valid. Please update your preferences in the app.</p>
      </body>
      </html>
    `);
    return;
  }

  const { userId } = decoded;

  // Get current preferences
  const preference = await prisma.emailPreference.findUnique({
    where: { userId },
  });

  const categories = (preference?.categories as Record<string, boolean>) || {};

  // Render unsubscribe form
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Email Preferences | Nutri</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 600px; margin: 50px auto; padding: 20px; }
        h1 { color: #1f2937; }
        .category { margin: 15px 0; padding: 10px; border: 1px solid #e5e7eb; border-radius: 8px; }
        .category label { display: flex; align-items: center; cursor: pointer; }
        .category input { margin-right: 10px; width: 20px; height: 20px; }
        button { background: #10b981; color: white; border: none; padding: 12px 24px;
                 border-radius: 8px; font-size: 16px; cursor: pointer; margin-top: 20px; }
        button:hover { background: #059669; }
        .unsubscribe-all { background: #ef4444; margin-left: 10px; }
        .unsubscribe-all:hover { background: #dc2626; }
      </style>
    </head>
    <body>
      <h1>Email Preferences</h1>
      <p>Choose which emails you'd like to receive:</p>

      <form action="/api/email/unsubscribe" method="POST">
        <input type="hidden" name="token" value="${token}">

        <div class="category">
          <label>
            <input type="checkbox" name="weekly_reports" ${categories.weekly_reports !== false ? 'checked' : ''}>
            Weekly Progress Reports
          </label>
        </div>

        <div class="category">
          <label>
            <input type="checkbox" name="health_insights" ${categories.health_insights !== false ? 'checked' : ''}>
            Health Insights
          </label>
        </div>

        <div class="category">
          <label>
            <input type="checkbox" name="tips" ${categories.tips !== false ? 'checked' : ''}>
            Tips & Recipes
          </label>
        </div>

        <div class="category">
          <label>
            <input type="checkbox" name="features" ${categories.features !== false ? 'checked' : ''}>
            Feature Updates
          </label>
        </div>

        <div class="category">
          <label>
            <input type="checkbox" name="promotions" ${categories.promotions === true ? 'checked' : ''}>
            Promotional Offers
          </label>
        </div>

        <button type="submit" name="action" value="save">Save Preferences</button>
        <button type="submit" name="action" value="unsubscribe_all" class="unsubscribe-all">
          Unsubscribe from All
        </button>
      </form>
    </body>
    </html>
  `);
}

/**
 * Process unsubscribe form
 * POST /api/email/unsubscribe
 */
export async function processUnsubscribe(req: Request, res: Response): Promise<void> {
  const { token, action, ...categories } = req.body;

  const decoded = verifyUnsubscribeToken(token);
  if (!decoded) {
    res.status(400).json({ error: 'Invalid or expired token' });
    return;
  }

  const { userId, campaignId } = decoded;

  if (action === 'unsubscribe_all') {
    // Unsubscribe from all marketing emails
    await prisma.emailPreference.upsert({
      where: { userId },
      update: { globalUnsubscribedAt: new Date(), marketingOptIn: false },
      create: { userId, globalUnsubscribedAt: new Date(), marketingOptIn: false },
    });

    // Exit any active sequences
    await prisma.emailSequenceEnrollment.updateMany({
      where: { userId, status: 'ACTIVE' },
      data: { status: 'EXITED', exitedAt: new Date(), exitReason: 'unsubscribed' },
    });

    logger.info({ userId, campaignId }, 'User unsubscribed from all marketing');

    res.send(`
      <!DOCTYPE html>
      <html>
      <head><title>Unsubscribed | Nutri</title></head>
      <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto; text-align: center;">
        <h1>You've been unsubscribed</h1>
        <p>You will no longer receive marketing emails from Nutri.</p>
        <p>You'll still receive important account notifications.</p>
      </body>
      </html>
    `);
    return;
  }

  // Save category preferences
  const categoryPreferences: Record<string, boolean> = {};
  for (const category of MARKETING_CATEGORIES) {
    categoryPreferences[category] = categories[category] === 'on';
  }

  await prisma.emailPreference.upsert({
    where: { userId },
    update: { categories: categoryPreferences, globalUnsubscribedAt: null },
    create: { userId, categories: categoryPreferences },
  });

  logger.info({ userId, categories: categoryPreferences }, 'Email preferences updated');

  res.send(`
    <!DOCTYPE html>
    <html>
    <head><title>Preferences Saved | Nutri</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto; text-align: center;">
      <h1>Preferences Saved</h1>
      <p>Your email preferences have been updated.</p>
    </body>
    </html>
  `);
}

/**
 * One-click unsubscribe (RFC 8058)
 * POST /api/email/unsubscribe/one-click
 */
export async function oneClickUnsubscribe(req: Request, res: Response): Promise<void> {
  // Check List-Unsubscribe-Post header
  const listUnsubscribePost = req.headers['list-unsubscribe-post'];
  if (listUnsubscribePost !== 'List-Unsubscribe=One-Click') {
    res.status(400).json({ error: 'Invalid one-click unsubscribe request' });
    return;
  }

  const { token } = req.query;
  if (typeof token !== 'string') {
    res.status(400).json({ error: 'Missing token' });
    return;
  }

  const decoded = verifyUnsubscribeToken(token);
  if (!decoded) {
    res.status(400).json({ error: 'Invalid token' });
    return;
  }

  const { userId, campaignId } = decoded;

  // Unsubscribe from all marketing
  await prisma.emailPreference.upsert({
    where: { userId },
    update: { globalUnsubscribedAt: new Date(), marketingOptIn: false },
    create: { userId, globalUnsubscribedAt: new Date(), marketingOptIn: false },
  });

  // Exit any active sequences
  await prisma.emailSequenceEnrollment.updateMany({
    where: { userId, status: 'ACTIVE' },
    data: { status: 'EXITED', exitedAt: new Date(), exitReason: 'one_click_unsubscribe' },
  });

  logger.info({ userId, campaignId }, 'One-click unsubscribe processed');

  // RFC 8058 requires 200 OK with no body
  res.status(200).send();
}

/**
 * Request double opt-in
 * POST /api/email/opt-in
 */
export async function requestDoubleOptIn(req: AuthenticatedRequest, res: Response): Promise<void> {
  const userId = req.userId;

  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  // Get or create preferences
  const preference = await prisma.emailPreference.upsert({
    where: { userId },
    update: {},
    create: { userId, marketingOptIn: false },
  });

  // If already confirmed, no need to send again
  if (preference.doubleOptInConfirmedAt) {
    res.json({ message: 'Already confirmed', confirmed: true });
    return;
  }

  // Send double opt-in email
  try {
    const token = generateUnsubscribeToken(userId);
    const confirmUrl = `${EMAIL_CONFIG.baseUrl}/api/email/opt-in/confirm?token=${encodeURIComponent(token)}`;

    await sendTransactional(userId, 'double-opt-in-confirmation', {
      confirmUrl,
    });

    res.json({ message: 'Confirmation email sent' });
  } catch (error) {
    logger.error({ userId, error }, 'Failed to send double opt-in email');
    res.status(500).json({ error: 'Failed to send confirmation email' });
  }
}

/**
 * Resubscribe (for users who previously unsubscribed)
 * POST /api/email/resubscribe
 */
export async function resubscribe(req: AuthenticatedRequest, res: Response): Promise<void> {
  const userId = req.userId;

  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  await prisma.emailPreference.update({
    where: { userId },
    data: {
      globalUnsubscribedAt: null,
      marketingOptIn: true,
      // Restore default categories
      categories: {
        weekly_reports: true,
        health_insights: true,
        tips: true,
        features: true,
        promotions: false,
        newsletter: false,
      },
    },
  });

  logger.info({ userId }, 'User resubscribed to marketing emails');

  res.json({ message: 'Successfully resubscribed' });
}

export default {
  getPreferences,
  updatePreferences,
  confirmDoubleOptIn,
  unsubscribePage,
  processUnsubscribe,
  oneClickUnsubscribe,
  requestDoubleOptIn,
  resubscribe,
};
