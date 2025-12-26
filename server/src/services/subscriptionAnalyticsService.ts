/**
 * Subscription Analytics Service
 *
 * Tracks subscription events, calculates metrics (MRR, LTV, churn),
 * and monitors refund patterns for alerting.
 */

import prisma from '../config/database';
import { logger } from '../config/logger';

/**
 * Subscription event types for analytics
 */
export type SubscriptionEventType =
  | 'paywall_viewed'
  | 'subscription_started'
  | 'trial_started'
  | 'trial_converted'
  | 'subscription_renewed'
  | 'subscription_canceled'
  | 'subscription_expired'
  | 'purchase_restored'
  | 'upgrade_completed'
  | 'downgrade_completed'
  | 'refund_processed'
  | 'billing_retry_started'
  | 'billing_retry_resolved';

/**
 * Subscription metrics for dashboard
 */
export interface SubscriptionMetrics {
  // Active subscriptions
  activeCount: number;
  trialCount: number;

  // Revenue
  mrr: number; // Monthly Recurring Revenue
  arr: number; // Annual Recurring Revenue

  // Rates
  conversionRate: number; // Trial → Paid
  churnRate: number; // Monthly churn
  refundRate: number; // Refunds / Total transactions

  // Averages
  averageLtv: number; // Lifetime Value
  averageSubscriptionLength: number; // In days

  // Growth
  newSubscriptions: number; // This period
  churned: number; // This period
  netGrowth: number;
}

/**
 * Refund alert threshold (percentage)
 */
const REFUND_ALERT_THRESHOLD = 0.05; // 5%

/**
 * Track a subscription event
 */
export async function trackEvent(
  eventType: SubscriptionEventType,
  userId: string,
  metadata?: Record<string, unknown>
): Promise<void> {
  try {
    logger.info(
      {
        event: eventType,
        userId,
        metadata,
      },
      'Subscription analytics event'
    );

    // TODO: Send to analytics service (Mixpanel, Amplitude, etc.)
    // For now, we just log. In production, integrate with:
    // - Mixpanel: mixpanel.track(eventType, { userId, ...metadata })
    // - Amplitude: amplitude.logEvent(eventType, { userId, ...metadata })
    // - Segment: analytics.track({ userId, event: eventType, properties: metadata })
  } catch (error) {
    logger.error({ error, eventType, userId }, 'Failed to track subscription event');
  }
}

/**
 * Calculate subscription metrics for a time period
 */
export async function calculateMetrics(
  startDate: Date,
  endDate: Date
): Promise<SubscriptionMetrics> {
  // Active subscriptions
  const activeSubscriptions = await prisma.subscription.count({
    where: {
      status: 'ACTIVE',
      expiresAt: { gte: new Date() },
    },
  });

  const trialSubscriptions = await prisma.subscription.count({
    where: {
      status: 'ACTIVE',
      isTrialPeriod: true,
    },
  });

  // New subscriptions in period
  const newSubscriptions = await prisma.subscription.count({
    where: {
      createdAt: { gte: startDate, lte: endDate },
    },
  });

  // Churned in period (expired or revoked)
  const churnedSubscriptions = await prisma.subscription.count({
    where: {
      OR: [{ status: 'EXPIRED' }, { status: 'REVOKED' }, { status: 'REFUNDED' }],
      updatedAt: { gte: startDate, lte: endDate },
    },
  });

  // Calculate MRR from active subscriptions
  const activeWithPricing = await prisma.subscription.findMany({
    where: {
      status: 'ACTIVE',
      expiresAt: { gte: new Date() },
      priceAmount: { not: null },
    },
    select: {
      productId: true,
      priceAmount: true,
    },
  });

  let mrr = 0;
  for (const sub of activeWithPricing) {
    const amount = sub.priceAmount ? Number(sub.priceAmount) : 0;
    // Convert yearly to monthly
    if (sub.productId.includes('yearly')) {
      mrr += amount / 12;
    } else {
      mrr += amount;
    }
  }

  // Trial conversion rate
  const trialsStarted = await prisma.subscriptionEvent.count({
    where: {
      notificationType: 'SUBSCRIBED',
      createdAt: { gte: startDate, lte: endDate },
      subscription: {
        isTrialPeriod: true,
      },
    },
  });

  const trialsConverted = await prisma.subscriptionEvent.count({
    where: {
      notificationType: 'DID_RENEW',
      createdAt: { gte: startDate, lte: endDate },
      subscription: {
        isTrialPeriod: false,
      },
    },
  });

  const conversionRate = trialsStarted > 0 ? trialsConverted / trialsStarted : 0;

  // Churn rate (churned / active at start of period)
  const activeAtStart = await prisma.subscription.count({
    where: {
      status: 'ACTIVE',
      createdAt: { lt: startDate },
    },
  });

  const churnRate = activeAtStart > 0 ? churnedSubscriptions / activeAtStart : 0;

  // Refund rate
  const refundEvents = await prisma.subscriptionEvent.count({
    where: {
      notificationType: 'REFUND',
      createdAt: { gte: startDate, lte: endDate },
    },
  });

  const totalTransactions = await prisma.subscriptionEvent.count({
    where: {
      createdAt: { gte: startDate, lte: endDate },
    },
  });

  const refundRate = totalTransactions > 0 ? refundEvents / totalTransactions : 0;

  // Check if refund rate exceeds threshold
  if (refundRate > REFUND_ALERT_THRESHOLD) {
    logger.warn(
      {
        refundRate,
        threshold: REFUND_ALERT_THRESHOLD,
        refundEvents,
        totalTransactions,
        period: { startDate, endDate },
      },
      'High refund rate detected - requires review'
    );
    // TODO: Send alert (email, Slack, PagerDuty, etc.)
  }

  // Average subscription length (for churned subscriptions)
  const churnedWithDates = await prisma.subscription.findMany({
    where: {
      OR: [{ status: 'EXPIRED' }, { status: 'REVOKED' }],
    },
    select: {
      createdAt: true,
      updatedAt: true,
    },
  });

  let totalDays = 0;
  for (const sub of churnedWithDates) {
    const days = (sub.updatedAt.getTime() - sub.createdAt.getTime()) / (1000 * 60 * 60 * 24);
    totalDays += days;
  }
  const averageSubscriptionLength =
    churnedWithDates.length > 0 ? totalDays / churnedWithDates.length : 0;

  // Average LTV (average subscription length * average monthly price)
  const averageMonthlyPrice = activeSubscriptions > 0 ? mrr / activeSubscriptions : 0;
  const averageLtv = (averageSubscriptionLength / 30) * averageMonthlyPrice;

  return {
    activeCount: activeSubscriptions,
    trialCount: trialSubscriptions,
    mrr,
    arr: mrr * 12,
    conversionRate,
    churnRate,
    refundRate,
    averageLtv,
    averageSubscriptionLength,
    newSubscriptions,
    churned: churnedSubscriptions,
    netGrowth: newSubscriptions - churnedSubscriptions,
  };
}

/**
 * Get subscription breakdown by product
 */
export async function getProductBreakdown(): Promise<
  Array<{
    productId: string;
    activeCount: number;
    revenue: number;
  }>
> {
  const subscriptions = await prisma.subscription.groupBy({
    by: ['productId'],
    where: {
      status: 'ACTIVE',
      expiresAt: { gte: new Date() },
    },
    _count: true,
    _sum: {
      priceAmount: true,
    },
  });

  return subscriptions.map((sub) => ({
    productId: sub.productId,
    activeCount: sub._count,
    revenue: sub._sum.priceAmount ? Number(sub._sum.priceAmount) : 0,
  }));
}

/**
 * Get recent subscription events for monitoring
 */
export async function getRecentEvents(limit: number = 50): Promise<
  Array<{
    id: string;
    type: string;
    subtype: string | null;
    userId: string | null;
    createdAt: Date;
  }>
> {
  const events = await prisma.subscriptionEvent.findMany({
    take: limit,
    orderBy: { createdAt: 'desc' },
    select: {
      id: true,
      notificationType: true,
      subtype: true,
      createdAt: true,
      subscription: {
        select: {
          userId: true,
        },
      },
    },
  });

  return events.map((event) => ({
    id: event.id,
    type: event.notificationType,
    subtype: event.subtype,
    userId: event.subscription?.userId ?? null,
    createdAt: event.createdAt,
  }));
}

/**
 * Check for high refund rate and alert if needed
 */
export async function checkRefundAlerts(): Promise<{
  isAlertTriggered: boolean;
  refundRate: number;
  message: string;
}> {
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

  const refunds = await prisma.subscriptionEvent.count({
    where: {
      notificationType: 'REFUND',
      createdAt: { gte: thirtyDaysAgo },
    },
  });

  const total = await prisma.subscriptionEvent.count({
    where: {
      notificationType: { in: ['SUBSCRIBED', 'DID_RENEW'] },
      createdAt: { gte: thirtyDaysAgo },
    },
  });

  const refundRate = total > 0 ? refunds / total : 0;
  const isAlertTriggered = refundRate > REFUND_ALERT_THRESHOLD;

  if (isAlertTriggered) {
    logger.error(
      {
        refundRate,
        refunds,
        total,
        threshold: REFUND_ALERT_THRESHOLD,
      },
      'ALERT: High refund rate detected'
    );
  }

  return {
    isAlertTriggered,
    refundRate,
    message: isAlertTriggered
      ? `High refund rate: ${(refundRate * 100).toFixed(1)}% (${refunds}/${total})`
      : `Refund rate normal: ${(refundRate * 100).toFixed(1)}%`,
  };
}

export default {
  trackEvent,
  calculateMetrics,
  getProductBreakdown,
  getRecentEvents,
  checkRefundAlerts,
};
