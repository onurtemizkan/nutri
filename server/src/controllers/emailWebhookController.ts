/**
 * Email Webhook Controller
 *
 * Handles email event webhooks from Resend.
 * Processes delivery, open, click, bounce, complaint, and unsubscribe events.
 */

import { Request, Response } from 'express';
import { z } from 'zod';
import prisma from '../config/database';
import { logger } from '../config/logger';
import { verifyResendWebhookSignature } from '../utils/emailHelpers';
import { trackEvent } from '../services/emailService';

/**
 * Resend webhook event types
 */
type ResendEventType =
  | 'email.sent'
  | 'email.delivered'
  | 'email.delivery_delayed'
  | 'email.complained'
  | 'email.bounced'
  | 'email.opened'
  | 'email.clicked';

/**
 * Webhook payload schema
 */
const webhookPayloadSchema = z.object({
  type: z.string(),
  created_at: z.string(),
  data: z.object({
    email_id: z.string(),
    from: z.string().optional(),
    to: z.array(z.string()).optional(),
    subject: z.string().optional(),
    created_at: z.string().optional(),
    // Bounce-specific fields
    bounce: z
      .object({
        message: z.string().optional(),
        type: z.string().optional(),
      })
      .optional(),
    // Click-specific fields
    click: z
      .object({
        link: z.string().optional(),
        timestamp: z.string().optional(),
        user_agent: z.string().optional(),
        ip_address: z.string().optional(),
      })
      .optional(),
    // Open-specific fields
    open: z
      .object({
        timestamp: z.string().optional(),
        user_agent: z.string().optional(),
        ip_address: z.string().optional(),
      })
      .optional(),
  }),
});

type WebhookPayload = z.infer<typeof webhookPayloadSchema>;

/**
 * Set to track processed webhook IDs (for idempotency)
 * In production, use Redis for distributed idempotency
 */
const processedWebhooks = new Set<string>();

/**
 * Handle Resend webhook
 */
export async function handleResendWebhook(req: Request, res: Response): Promise<void> {
  const signature = req.headers['svix-signature'] as string | undefined;
  const svixId = req.headers['svix-id'] as string | undefined;
  const rawBody = JSON.stringify(req.body);

  // Log incoming webhook
  logger.info(
    {
      svixId,
      hasSignature: !!signature,
    },
    'Received email webhook'
  );

  // Verify signature
  if (signature && !verifyResendWebhookSignature(rawBody, signature)) {
    logger.warn({ svixId }, 'Invalid webhook signature');
    res.status(401).json({ error: 'Invalid signature' });
    return;
  }

  // Check for duplicate (idempotency)
  if (svixId && processedWebhooks.has(svixId)) {
    logger.info({ svixId }, 'Duplicate webhook, already processed');
    res.status(200).json({ message: 'Already processed' });
    return;
  }

  // Parse and validate payload
  let payload: WebhookPayload;
  try {
    payload = webhookPayloadSchema.parse(req.body);
  } catch (error) {
    logger.error({ error, body: req.body }, 'Invalid webhook payload');
    res.status(400).json({ error: 'Invalid payload' });
    return;
  }

  // Mark as processed
  if (svixId) {
    processedWebhooks.add(svixId);
    // Clean up old entries after 24 hours (in production, use Redis with TTL)
    setTimeout(() => processedWebhooks.delete(svixId), 24 * 60 * 60 * 1000);
  }

  // Respond immediately
  res.status(200).json({ received: true });

  // Process event asynchronously
  try {
    await processWebhookEvent(payload);
  } catch (error) {
    logger.error({ error, payload }, 'Error processing webhook event');
  }
}

/**
 * Process webhook event
 */
async function processWebhookEvent(payload: WebhookPayload): Promise<void> {
  const { type, data } = payload;
  const emailId = data.email_id;

  logger.info({ type, emailId }, 'Processing email webhook event');

  switch (type as ResendEventType) {
    case 'email.sent':
      await handleSent(emailId);
      break;
    case 'email.delivered':
      await handleDelivered(emailId);
      break;
    case 'email.delivery_delayed':
      await handleDeliveryDelayed(emailId);
      break;
    case 'email.bounced':
      await handleBounce(emailId, data.bounce);
      break;
    case 'email.complained':
      await handleComplaint(emailId);
      break;
    case 'email.opened':
      await handleOpen(emailId, data.open);
      break;
    case 'email.clicked':
      await handleClick(emailId, data.click);
      break;
    default:
      logger.warn({ type }, 'Unknown webhook event type');
  }
}

/**
 * Handle email sent event
 */
async function handleSent(emailId: string): Promise<void> {
  await trackEvent(emailId, 'sent');
}

/**
 * Handle email delivered event
 */
async function handleDelivered(emailId: string): Promise<void> {
  await trackEvent(emailId, 'delivered');
}

/**
 * Handle delivery delayed event
 */
async function handleDeliveryDelayed(emailId: string): Promise<void> {
  logger.warn({ emailId }, 'Email delivery delayed');
  // Could notify admin or update status
}

/**
 * Handle bounce event
 */
async function handleBounce(
  emailId: string,
  bounceData?: { message?: string; type?: string }
): Promise<void> {
  await trackEvent(emailId, 'bounced', {
    bounceType: bounceData?.type || 'unknown',
    bounceMessage: bounceData?.message,
  });

  // If hard bounce, add to suppression list
  if (bounceData?.type === 'hard') {
    const emailLog = await prisma.emailLog.findFirst({
      where: { providerId: emailId },
      select: { email: true, userId: true },
    });

    if (emailLog?.userId) {
      // Update user's email preference to mark as undeliverable
      await prisma.emailPreference.upsert({
        where: { userId: emailLog.userId },
        update: {
          globalUnsubscribedAt: new Date(),
        },
        create: {
          userId: emailLog.userId,
          globalUnsubscribedAt: new Date(),
        },
      });

      logger.warn(
        { email: emailLog.email, userId: emailLog.userId },
        'Hard bounce - user added to suppression list'
      );
    }
  }
}

/**
 * Handle complaint event (spam report)
 */
async function handleComplaint(emailId: string): Promise<void> {
  await trackEvent(emailId, 'complained');

  // Get the email log to find the user
  const emailLog = await prisma.emailLog.findFirst({
    where: { providerId: emailId },
    select: { email: true, userId: true },
  });

  if (emailLog?.userId) {
    // Auto-unsubscribe from all marketing
    await prisma.emailPreference.upsert({
      where: { userId: emailLog.userId },
      update: {
        globalUnsubscribedAt: new Date(),
        marketingOptIn: false,
      },
      create: {
        userId: emailLog.userId,
        globalUnsubscribedAt: new Date(),
        marketingOptIn: false,
      },
    });

    // Exit any active sequences
    await prisma.emailSequenceEnrollment.updateMany({
      where: {
        userId: emailLog.userId,
        status: 'ACTIVE',
      },
      data: {
        status: 'EXITED',
        exitedAt: new Date(),
        exitReason: 'complaint',
      },
    });

    logger.warn(
      { email: emailLog.email, userId: emailLog.userId },
      'Complaint received - user unsubscribed from all marketing'
    );
  }

  // Check refund rate and alert if high
  await checkComplaintRate();
}

/**
 * Handle open event
 */
async function handleOpen(
  emailId: string,
  openData?: { timestamp?: string; user_agent?: string; ip_address?: string }
): Promise<void> {
  // Detect Apple Mail Privacy Protection
  const isAppleProxy =
    openData?.user_agent?.includes('Apple') &&
    (openData?.ip_address?.startsWith('17.') || // Apple's IP range
      openData?.ip_address?.startsWith('104.'));

  await trackEvent(emailId, 'opened', {
    userAgent: openData?.user_agent,
    ipAddress: openData?.ip_address,
    isAppleProxy,
  });
}

/**
 * Handle click event
 */
async function handleClick(
  emailId: string,
  clickData?: { link?: string; timestamp?: string; user_agent?: string; ip_address?: string }
): Promise<void> {
  await trackEvent(emailId, 'clicked', {
    url: clickData?.link,
    userAgent: clickData?.user_agent,
    ipAddress: clickData?.ip_address,
  });
}

/**
 * Check complaint rate and alert if too high
 */
async function checkComplaintRate(): Promise<void> {
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

  const [complaints, totalSent] = await Promise.all([
    prisma.emailLog.count({
      where: {
        status: 'COMPLAINED',
        createdAt: { gte: thirtyDaysAgo },
      },
    }),
    prisma.emailLog.count({
      where: {
        status: { in: ['SENT', 'DELIVERED', 'OPENED', 'CLICKED'] },
        createdAt: { gte: thirtyDaysAgo },
      },
    }),
  ]);

  const complaintRate = totalSent > 0 ? complaints / totalSent : 0;
  const threshold = 0.001; // 0.1% - industry standard

  if (complaintRate > threshold) {
    logger.error(
      {
        complaintRate,
        threshold,
        complaints,
        totalSent,
      },
      'HIGH COMPLAINT RATE ALERT - exceeds industry threshold'
    );
    // TODO: Send alert to admin (Slack, email, PagerDuty)
  }
}

/**
 * Get email webhook statistics
 */
export async function getWebhookStats(_req: Request, res: Response): Promise<void> {
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

  const [stats, bouncesByType, recentEvents] = await Promise.all([
    // Overall stats
    prisma.emailLog.groupBy({
      by: ['status'],
      where: { createdAt: { gte: thirtyDaysAgo } },
      _count: true,
    }),

    // Bounces by type
    prisma.emailLog.groupBy({
      by: ['bounceType'],
      where: {
        status: 'BOUNCED',
        createdAt: { gte: thirtyDaysAgo },
      },
      _count: true,
    }),

    // Recent events
    prisma.emailLog.findMany({
      where: { createdAt: { gte: thirtyDaysAgo } },
      orderBy: { createdAt: 'desc' },
      take: 20,
      select: {
        id: true,
        email: true,
        status: true,
        templateSlug: true,
        createdAt: true,
        sentAt: true,
        deliveredAt: true,
        openedAt: true,
        clickedAt: true,
        bouncedAt: true,
      },
    }),
  ]);

  res.json({
    period: 'last_30_days',
    stats: stats.map((s) => ({ status: s.status, count: s._count })),
    bouncesByType: bouncesByType.map((b) => ({
      type: b.bounceType || 'unknown',
      count: b._count,
    })),
    recentEvents,
  });
}

export default {
  handleResendWebhook,
  getWebhookStats,
};
