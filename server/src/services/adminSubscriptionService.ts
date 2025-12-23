import prisma from '../config/database';
import { Prisma, SubscriptionTier, BillingCycle } from '@prisma/client';
import { logger } from '../config/logger';

// Types
export interface ListSubscriptionsParams {
  status?: 'active' | 'trial' | 'expired' | 'none';
  page: number;
  limit: number;
}

export interface SubscriptionListItem {
  id: string;
  email: string;
  name: string;
  subscriptionTier: SubscriptionTier;
  subscriptionBillingCycle: BillingCycle | null;
  subscriptionStartDate: Date | null;
  subscriptionEndDate: Date | null;
  subscriptionPrice: number | null;
  createdAt: Date;
}

export interface PaginatedSubscriptionsResponse {
  subscriptions: SubscriptionListItem[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface SubscriptionDetailResponse {
  id: string;
  email: string;
  name: string;
  subscriptionTier: SubscriptionTier;
  subscriptionBillingCycle: BillingCycle | null;
  subscriptionStartDate: Date | null;
  subscriptionEndDate: Date | null;
  subscriptionPrice: number | null;
  appleId: string | null;
  createdAt: Date;
  updatedAt: Date;
  isActive: boolean;
  daysRemaining: number | null;
}

type GrantDuration = '7_days' | '30_days' | '90_days' | '1_year';

/**
 * Get paginated list of users with subscriptions
 */
export async function getSubscriptionList(
  params: ListSubscriptionsParams
): Promise<PaginatedSubscriptionsResponse> {
  const { status, page, limit } = params;

  // Build where clause
  const where: Prisma.UserWhereInput = {};
  const now = new Date();

  // Filter by subscription status
  if (status) {
    switch (status) {
      case 'active':
        where.subscriptionTier = 'PRO';
        where.subscriptionEndDate = { gt: now };
        break;
      case 'trial':
        where.subscriptionTier = 'PRO_TRIAL';
        where.subscriptionEndDate = { gt: now };
        break;
      case 'expired':
        where.subscriptionEndDate = { lt: now };
        where.NOT = { subscriptionTier: 'FREE' };
        break;
      case 'none':
        where.subscriptionTier = 'FREE';
        break;
    }
  } else {
    // Exclude FREE users when listing subscriptions
    where.NOT = { subscriptionTier: 'FREE' };
  }

  const skip = (page - 1) * limit;

  const [users, total] = await Promise.all([
    prisma.user.findMany({
      where,
      skip,
      take: limit,
      orderBy: { subscriptionEndDate: 'desc' },
      select: {
        id: true,
        email: true,
        name: true,
        subscriptionTier: true,
        subscriptionBillingCycle: true,
        subscriptionStartDate: true,
        subscriptionEndDate: true,
        subscriptionPrice: true,
        createdAt: true,
      },
    }),
    prisma.user.count({ where }),
  ]);

  const totalPages = Math.ceil(total / limit);

  return {
    subscriptions: users,
    pagination: {
      page,
      limit,
      total,
      totalPages,
    },
  };
}

/**
 * Get detailed subscription information for a user
 */
export async function getSubscriptionDetail(
  userId: string
): Promise<SubscriptionDetailResponse | null> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      email: true,
      name: true,
      subscriptionTier: true,
      subscriptionBillingCycle: true,
      subscriptionStartDate: true,
      subscriptionEndDate: true,
      subscriptionPrice: true,
      appleId: true,
      createdAt: true,
      updatedAt: true,
    },
  });

  if (!user) {
    return null;
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
    ...user,
    isActive,
    daysRemaining,
  };
}

/**
 * Lookup user by Apple transaction ID (stored in appleId field)
 */
export async function lookupByTransactionId(
  transactionId: string
): Promise<SubscriptionDetailResponse | null> {
  // First try to find by appleId
  const user = await prisma.user.findFirst({
    where: {
      OR: [
        { appleId: transactionId },
        // Could also search in a metadata field if we stored transaction IDs there
      ],
    },
    select: {
      id: true,
      email: true,
      name: true,
      subscriptionTier: true,
      subscriptionBillingCycle: true,
      subscriptionStartDate: true,
      subscriptionEndDate: true,
      subscriptionPrice: true,
      appleId: true,
      createdAt: true,
      updatedAt: true,
    },
  });

  if (!user) {
    return null;
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
    ...user,
    isActive,
    daysRemaining,
  };
}

/**
 * Manually grant Pro subscription to a user
 */
export async function grantSubscription(
  userId: string,
  duration: GrantDuration,
  reason: string,
  adminUserId: string
): Promise<SubscriptionDetailResponse | null> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
  });

  if (!user) {
    return null;
  }

  // Calculate expiration date based on duration
  const now = new Date();
  let expiresAt: Date;

  switch (duration) {
    case '7_days':
      expiresAt = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
      break;
    case '30_days':
      expiresAt = new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
      break;
    case '90_days':
      expiresAt = new Date(now.getTime() + 90 * 24 * 60 * 60 * 1000);
      break;
    case '1_year':
      expiresAt = new Date(now.getTime() + 365 * 24 * 60 * 60 * 1000);
      break;
  }

  const updated = await prisma.user.update({
    where: { id: userId },
    data: {
      subscriptionTier: 'PRO',
      subscriptionStartDate: now,
      subscriptionEndDate: expiresAt,
      subscriptionBillingCycle: null, // Manual grants don't have billing cycles
      subscriptionPrice: 0, // Free manual grant
    },
    select: {
      id: true,
      email: true,
      name: true,
      subscriptionTier: true,
      subscriptionBillingCycle: true,
      subscriptionStartDate: true,
      subscriptionEndDate: true,
      subscriptionPrice: true,
      appleId: true,
      createdAt: true,
      updatedAt: true,
    },
  });

  logger.info(
    {
      userId,
      adminUserId,
      duration,
      reason,
      expiresAt,
    },
    'Subscription granted manually'
  );

  return {
    ...updated,
    isActive: true,
    daysRemaining: Math.ceil(
      (expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
    ),
  };
}

/**
 * Extend an existing subscription
 */
export async function extendSubscription(
  userId: string,
  days: number,
  reason: string,
  adminUserId: string
): Promise<SubscriptionDetailResponse | null> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
  });

  if (!user) {
    return null;
  }

  // Calculate new expiration date
  const now = new Date();
  const currentEnd = user.subscriptionEndDate || now;
  const baseDate = currentEnd > now ? currentEnd : now;
  const newEndDate = new Date(baseDate.getTime() + days * 24 * 60 * 60 * 1000);

  const updated = await prisma.user.update({
    where: { id: userId },
    data: {
      subscriptionEndDate: newEndDate,
      // Upgrade to PRO if currently FREE or trial
      subscriptionTier:
        user.subscriptionTier === 'FREE' ? 'PRO' : user.subscriptionTier,
      subscriptionStartDate: user.subscriptionStartDate || now,
    },
    select: {
      id: true,
      email: true,
      name: true,
      subscriptionTier: true,
      subscriptionBillingCycle: true,
      subscriptionStartDate: true,
      subscriptionEndDate: true,
      subscriptionPrice: true,
      appleId: true,
      createdAt: true,
      updatedAt: true,
    },
  });

  logger.info(
    {
      userId,
      adminUserId,
      days,
      reason,
      previousEndDate: currentEnd,
      newEndDate,
    },
    'Subscription extended'
  );

  return {
    ...updated,
    isActive: true,
    daysRemaining: Math.ceil(
      (newEndDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
    ),
  };
}

/**
 * Revoke a subscription
 */
export async function revokeSubscription(
  userId: string,
  reason: string,
  adminUserId: string
): Promise<SubscriptionDetailResponse | null> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
  });

  if (!user) {
    return null;
  }

  const now = new Date();

  const updated = await prisma.user.update({
    where: { id: userId },
    data: {
      subscriptionTier: 'FREE',
      subscriptionEndDate: now,
      // Keep start date and billing cycle for records
    },
    select: {
      id: true,
      email: true,
      name: true,
      subscriptionTier: true,
      subscriptionBillingCycle: true,
      subscriptionStartDate: true,
      subscriptionEndDate: true,
      subscriptionPrice: true,
      appleId: true,
      createdAt: true,
      updatedAt: true,
    },
  });

  logger.warn(
    {
      userId,
      userEmail: user.email,
      adminUserId,
      reason,
      previousTier: user.subscriptionTier,
    },
    'Subscription revoked'
  );

  return {
    ...updated,
    isActive: false,
    daysRemaining: 0,
  };
}
