/**
 * Subscription Service
 *
 * Handles subscription management for App Store IAP with StoreKit 2 support.
 * Manages subscription lifecycle: create, update, renew, expire, revoke.
 */

import prisma from '../config/database';
import { Prisma, SubscriptionStatus, SubscriptionTier } from '@prisma/client';
import { logger } from '../config/logger';

// ============================================================================
// Types
// ============================================================================

export interface CreateSubscriptionInput {
  userId: string;
  originalTransactionId: string;
  productId: string;
  purchaseDate: Date;
  expiresDate: Date | null;
  environment: 'Production' | 'Sandbox';
  isTrialPeriod?: boolean;
  isIntroOfferPeriod?: boolean;
  autoRenewEnabled?: boolean;
  autoRenewProductId?: string;
  priceLocale?: string;
  priceCurrency?: string;
  priceAmount?: number;
}

export interface UpdateSubscriptionInput {
  status?: SubscriptionStatus;
  expiresAt?: Date | null;
  cancelledAt?: Date | null;
  autoRenewEnabled?: boolean;
  autoRenewProductId?: string | null;
  gracePeriodExpiresAt?: Date | null;
  billingRetryExpiresAt?: Date | null;
}

export interface SubscriptionStatusResponse {
  isActive: boolean;
  tier: SubscriptionTier;
  status: SubscriptionStatus | null;
  expiresAt: Date | null;
  daysRemaining: number | null;
  isTrialPeriod: boolean;
  autoRenewEnabled: boolean;
  productId: string | null;
}

export interface CreateSubscriptionEventInput {
  subscriptionId: string;
  notificationType: string;
  subtype?: string;
  transactionId?: string;
  originalTransactionId: string;
  notificationUUID?: string;
  eventData?: Record<string, unknown>;
}

// ============================================================================
// Product Configuration
// ============================================================================

export const PRODUCT_IDS = {
  PRO_MONTHLY: 'nutri.pro.monthly',
  PRO_YEARLY: 'nutri.pro.yearly',
} as const;

export type ProductId = (typeof PRODUCT_IDS)[keyof typeof PRODUCT_IDS];

/**
 * Map product ID to subscription tier
 */
function getSubscriptionTierFromProduct(productId: string): SubscriptionTier {
  if (productId.includes('pro') || productId.includes('premium')) {
    return 'PRO';
  }
  return 'FREE';
}

/**
 * Map product ID to billing cycle
 */
function getBillingCycleFromProduct(productId: string): 'MONTHLY' | 'ANNUAL' | null {
  if (productId.includes('yearly') || productId.includes('annual')) {
    return 'ANNUAL';
  }
  if (productId.includes('monthly')) {
    return 'MONTHLY';
  }
  return null;
}

// ============================================================================
// Subscription Service
// ============================================================================

/**
 * Create or update a subscription from a purchase
 */
export async function createOrUpdateSubscription(
  input: CreateSubscriptionInput
): Promise<{ subscription: Prisma.SubscriptionGetPayload<object>; isNew: boolean }> {
  const {
    userId,
    originalTransactionId,
    productId,
    purchaseDate,
    expiresDate,
    environment,
    isTrialPeriod = false,
    isIntroOfferPeriod = false,
    autoRenewEnabled = true,
    autoRenewProductId,
    priceLocale,
    priceCurrency,
    priceAmount,
  } = input;

  // Check if subscription already exists
  const existing = await prisma.subscription.findUnique({
    where: { originalTransactionId },
  });

  const subscriptionData: Prisma.SubscriptionCreateInput | Prisma.SubscriptionUpdateInput = {
    productId,
    status: 'ACTIVE',
    expiresAt: expiresDate || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
    isTrialPeriod,
    isIntroOfferPeriod,
    autoRenewEnabled,
    autoRenewProductId,
    priceLocale,
    priceCurrency,
    priceAmount: priceAmount ? new Prisma.Decimal(priceAmount) : null,
    environment: environment === 'Production' ? 'PRODUCTION' : 'SANDBOX',
  };

  let subscription;
  let isNew = false;

  if (existing) {
    // Update existing subscription
    subscription = await prisma.subscription.update({
      where: { id: existing.id },
      data: subscriptionData,
    });

    logger.info(
      {
        subscriptionId: subscription.id,
        originalTransactionId,
        productId,
      },
      'Updated existing subscription'
    );
  } else {
    // Create new subscription
    subscription = await prisma.subscription.create({
      data: {
        ...(subscriptionData as Prisma.SubscriptionCreateInput),
        user: { connect: { id: userId } },
        originalTransactionId,
      },
    });
    isNew = true;

    logger.info(
      {
        subscriptionId: subscription.id,
        userId,
        originalTransactionId,
        productId,
      },
      'Created new subscription'
    );
  }

  // Update user subscription fields
  await updateUserSubscriptionFields(userId, {
    tier: isTrialPeriod ? 'PRO_TRIAL' : getSubscriptionTierFromProduct(productId),
    startDate: purchaseDate,
    endDate: expiresDate || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
    billingCycle: getBillingCycleFromProduct(productId),
    price: priceAmount || 0,
  });

  return { subscription, isNew };
}

/**
 * Update user's subscription fields on the User model
 */
async function updateUserSubscriptionFields(
  userId: string,
  data: {
    tier: SubscriptionTier;
    startDate: Date;
    endDate: Date;
    billingCycle: 'MONTHLY' | 'ANNUAL' | null;
    price: number;
  }
): Promise<void> {
  await prisma.user.update({
    where: { id: userId },
    data: {
      subscriptionTier: data.tier,
      subscriptionStartDate: data.startDate,
      subscriptionEndDate: data.endDate,
      subscriptionBillingCycle: data.billingCycle,
      subscriptionPrice: data.price,
    },
  });
}

/**
 * Find subscription by original transaction ID
 */
export async function findByTransactionId(
  originalTransactionId: string
): Promise<Prisma.SubscriptionGetPayload<{ include: { user: true } }> | null> {
  return prisma.subscription.findUnique({
    where: { originalTransactionId },
    include: { user: true },
  });
}

/**
 * Find subscription by user ID
 */
export async function findByUserId(
  userId: string
): Promise<Prisma.SubscriptionGetPayload<object> | null> {
  return prisma.subscription.findUnique({
    where: { userId },
  });
}

/**
 * Update subscription status
 */
export async function updateSubscription(
  originalTransactionId: string,
  data: UpdateSubscriptionInput
): Promise<Prisma.SubscriptionGetPayload<{ include: { user: true } }> | null> {
  const subscription = await prisma.subscription.findUnique({
    where: { originalTransactionId },
    include: { user: true },
  });

  if (!subscription) {
    logger.warn({ originalTransactionId }, 'Subscription not found for update');
    return null;
  }

  const updated = await prisma.subscription.update({
    where: { id: subscription.id },
    data: {
      status: data.status,
      expiresAt: data.expiresAt ?? undefined,
      cancelledAt: data.cancelledAt,
      autoRenewEnabled: data.autoRenewEnabled,
      autoRenewProductId: data.autoRenewProductId,
      gracePeriodExpiresAt: data.gracePeriodExpiresAt,
      billingRetryExpiresAt: data.billingRetryExpiresAt,
    },
    include: { user: true },
  });

  // If status changed to expired or revoked, update user tier
  if (data.status === 'EXPIRED' || data.status === 'REVOKED' || data.status === 'REFUNDED') {
    await prisma.user.update({
      where: { id: subscription.userId },
      data: {
        subscriptionTier: 'FREE',
      },
    });

    logger.info(
      {
        userId: subscription.userId,
        originalTransactionId,
        newStatus: data.status,
      },
      'User downgraded to FREE tier'
    );
  }

  // If subscription renewed, update user end date
  if (data.expiresAt && data.status === 'ACTIVE') {
    await prisma.user.update({
      where: { id: subscription.userId },
      data: {
        subscriptionEndDate: data.expiresAt,
      },
    });
  }

  return updated;
}

/**
 * Handle subscription renewal
 */
export async function renewSubscription(
  originalTransactionId: string,
  newExpiresDate: Date
): Promise<Prisma.SubscriptionGetPayload<object> | null> {
  return updateSubscription(originalTransactionId, {
    status: 'ACTIVE',
    expiresAt: newExpiresDate,
    gracePeriodExpiresAt: null,
    billingRetryExpiresAt: null,
  });
}

/**
 * Handle subscription expiration
 */
export async function expireSubscription(
  originalTransactionId: string
): Promise<Prisma.SubscriptionGetPayload<object> | null> {
  return updateSubscription(originalTransactionId, {
    status: 'EXPIRED',
  });
}

/**
 * Handle subscription revocation (refund)
 */
export async function revokeSubscription(
  originalTransactionId: string,
  reason: 'REFUND' | 'OTHER' = 'OTHER'
): Promise<Prisma.SubscriptionGetPayload<object> | null> {
  return updateSubscription(originalTransactionId, {
    status: reason === 'REFUND' ? 'REFUNDED' : 'REVOKED',
    cancelledAt: new Date(),
  });
}

/**
 * Handle grace period entry
 */
export async function enterGracePeriod(
  originalTransactionId: string,
  gracePeriodExpiresDate: Date
): Promise<Prisma.SubscriptionGetPayload<object> | null> {
  return updateSubscription(originalTransactionId, {
    status: 'IN_GRACE_PERIOD',
    gracePeriodExpiresAt: gracePeriodExpiresDate,
  });
}

/**
 * Handle billing retry period entry
 */
export async function enterBillingRetry(
  originalTransactionId: string
): Promise<Prisma.SubscriptionGetPayload<object> | null> {
  return updateSubscription(originalTransactionId, {
    status: 'IN_BILLING_RETRY',
  });
}

/**
 * Get subscription status for a user
 */
export async function getSubscriptionStatus(userId: string): Promise<SubscriptionStatusResponse> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      subscriptionTier: true,
      subscriptionEndDate: true,
      subscription: {
        select: {
          status: true,
          expiresAt: true,
          isTrialPeriod: true,
          autoRenewEnabled: true,
          productId: true,
        },
      },
    },
  });

  if (!user) {
    return {
      isActive: false,
      tier: 'FREE',
      status: null,
      expiresAt: null,
      daysRemaining: null,
      isTrialPeriod: false,
      autoRenewEnabled: false,
      productId: null,
    };
  }

  const now = new Date();
  const isActive =
    user.subscriptionTier !== 'FREE' &&
    user.subscriptionEndDate !== null &&
    user.subscriptionEndDate > now;

  let daysRemaining: number | null = null;
  if (user.subscriptionEndDate) {
    const diff = user.subscriptionEndDate.getTime() - now.getTime();
    daysRemaining = Math.max(0, Math.ceil(diff / (1000 * 60 * 60 * 24)));
  }

  return {
    isActive,
    tier: user.subscriptionTier,
    status: user.subscription?.status ?? null,
    expiresAt: user.subscriptionEndDate,
    daysRemaining,
    isTrialPeriod: user.subscription?.isTrialPeriod ?? false,
    autoRenewEnabled: user.subscription?.autoRenewEnabled ?? false,
    productId: user.subscription?.productId ?? null,
  };
}

/**
 * Create a subscription event audit log
 */
export async function createSubscriptionEvent(
  input: CreateSubscriptionEventInput
): Promise<Prisma.SubscriptionEventGetPayload<object>> {
  const event = await prisma.subscriptionEvent.create({
    data: {
      subscriptionId: input.subscriptionId,
      notificationType: input.notificationType,
      subtype: input.subtype,
      transactionId: input.transactionId,
      originalTransactionId: input.originalTransactionId,
      notificationUUID: input.notificationUUID,
      eventData: input.eventData as Prisma.InputJsonValue,
    },
  });

  logger.info(
    {
      eventId: event.id,
      notificationType: input.notificationType,
      originalTransactionId: input.originalTransactionId,
    },
    'Created subscription event'
  );

  return event;
}

/**
 * Restore purchases for a user
 * Verifies and restores active subscriptions from transaction IDs
 */
export async function restorePurchases(
  userId: string,
  transactionIds: string[]
): Promise<{
  restored: number;
  alreadyActive: number;
  errors: string[];
}> {
  let restored = 0;
  let alreadyActive = 0;
  const errors: string[] = [];

  for (const transactionId of transactionIds) {
    try {
      const subscription = await prisma.subscription.findUnique({
        where: { originalTransactionId: transactionId },
      });

      if (!subscription) {
        errors.push(`Transaction ${transactionId} not found`);
        continue;
      }

      // Check if subscription is still valid
      const now = new Date();
      if (subscription.expiresAt < now) {
        errors.push(`Transaction ${transactionId} has expired`);
        continue;
      }

      // Check if already linked to this user
      if (subscription.userId === userId) {
        alreadyActive++;
        continue;
      }

      // This is a restoration from a different device/reinstall
      // Update the user's subscription fields
      await updateUserSubscriptionFields(userId, {
        tier: subscription.isTrialPeriod
          ? 'PRO_TRIAL'
          : getSubscriptionTierFromProduct(subscription.productId),
        startDate: new Date(),
        endDate: subscription.expiresAt,
        billingCycle: getBillingCycleFromProduct(subscription.productId),
        price: subscription.priceAmount?.toNumber() ?? 0,
      });

      restored++;

      logger.info(
        {
          userId,
          transactionId,
          expiresAt: subscription.expiresAt,
        },
        'Restored subscription'
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      errors.push(`Failed to restore ${transactionId}: ${message}`);
      logger.error({ error, transactionId }, 'Failed to restore subscription');
    }
  }

  return { restored, alreadyActive, errors };
}

/**
 * Check and expire stale subscriptions
 * Run this as a scheduled job
 */
export async function expireStaleSubscriptions(): Promise<number> {
  const now = new Date();

  // Find subscriptions that are past their expiry but still marked active
  const staleSubscriptions = await prisma.subscription.findMany({
    where: {
      status: 'ACTIVE',
      expiresAt: { lt: now },
      gracePeriodExpiresAt: null, // Not in grace period
    },
    include: { user: true },
  });

  let expired = 0;

  for (const subscription of staleSubscriptions) {
    try {
      await expireSubscription(subscription.originalTransactionId);
      expired++;

      logger.info(
        {
          subscriptionId: subscription.id,
          userId: subscription.userId,
          originalTransactionId: subscription.originalTransactionId,
        },
        'Expired stale subscription'
      );
    } catch (error) {
      logger.error(
        { error, subscriptionId: subscription.id },
        'Failed to expire stale subscription'
      );
    }
  }

  if (expired > 0) {
    logger.info({ expired }, 'Expired stale subscriptions');
  }

  return expired;
}

export default {
  createOrUpdateSubscription,
  findByTransactionId,
  findByUserId,
  updateSubscription,
  renewSubscription,
  expireSubscription,
  revokeSubscription,
  enterGracePeriod,
  enterBillingRetry,
  getSubscriptionStatus,
  createSubscriptionEvent,
  restorePurchases,
  expireStaleSubscriptions,
  PRODUCT_IDS,
};
