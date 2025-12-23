import prisma from '../config/database';
import { Prisma, SubscriptionTier } from '@prisma/client';
import { logger } from '../config/logger';

// Types
export interface ListUsersParams {
  search?: string;
  page: number;
  limit: number;
  sortBy: 'createdAt' | 'email' | 'name';
  sortOrder: 'asc' | 'desc';
  subscriptionStatus?: 'active' | 'trial' | 'expired' | 'none';
}

export interface UserListItem {
  id: string;
  email: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
  subscriptionTier: SubscriptionTier;
  subscriptionEndDate: Date | null;
  subscriptionStartDate: Date | null;
  _count: {
    meals: number;
    healthMetrics: number;
    activities: number;
  };
}

export interface PaginatedUsersResponse {
  users: UserListItem[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface UserDetailResponse {
  id: string;
  email: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
  profilePicture: string | null;
  goalCalories: number;
  goalProtein: number;
  goalCarbs: number;
  goalFat: number;
  currentWeight: number | null;
  goalWeight: number | null;
  height: number | null;
  activityLevel: string;
  subscriptionTier: SubscriptionTier;
  subscriptionBillingCycle: string | null;
  subscriptionStartDate: Date | null;
  subscriptionEndDate: Date | null;
  subscriptionPrice: number | null;
  appleId: string | null;
  _count: {
    meals: number;
    healthMetrics: number;
    activities: number;
    supplements: number;
  };
  recentActivity: {
    mealsLast7Days: number;
    healthMetricsLast7Days: number;
    activitiesLast7Days: number;
  };
}

/**
 * Get paginated list of users with search and filters
 */
export async function getUserList(
  params: ListUsersParams
): Promise<PaginatedUsersResponse> {
  const { search, page, limit, sortBy, sortOrder, subscriptionStatus } = params;

  // Build where clause
  const where: Prisma.UserWhereInput = {};

  // Search filter (email or name)
  if (search && search.trim()) {
    where.OR = [
      { email: { contains: search, mode: 'insensitive' } },
      { name: { contains: search, mode: 'insensitive' } },
    ];
  }

  // Subscription status filter
  if (subscriptionStatus) {
    const now = new Date();

    switch (subscriptionStatus) {
      case 'active':
        // PRO tier with valid subscription
        where.subscriptionTier = 'PRO';
        where.subscriptionEndDate = { gt: now };
        break;
      case 'trial':
        // PRO_TRIAL tier with valid trial
        where.subscriptionTier = 'PRO_TRIAL';
        where.subscriptionEndDate = { gt: now };
        break;
      case 'expired':
        // Any tier with expired subscription
        where.subscriptionEndDate = { lt: now };
        break;
      case 'none':
        // FREE tier only
        where.subscriptionTier = 'FREE';
        break;
    }
  }

  // Calculate offset
  const skip = (page - 1) * limit;

  // Build order by
  const orderBy: Prisma.UserOrderByWithRelationInput = {
    [sortBy]: sortOrder,
  };

  // Execute queries in parallel
  const [users, total] = await Promise.all([
    prisma.user.findMany({
      where,
      skip,
      take: limit,
      orderBy,
      select: {
        id: true,
        email: true,
        name: true,
        createdAt: true,
        updatedAt: true,
        subscriptionTier: true,
        subscriptionEndDate: true,
        subscriptionStartDate: true,
        _count: {
          select: {
            meals: true,
            healthMetrics: true,
            activities: true,
          },
        },
      },
    }),
    prisma.user.count({ where }),
  ]);

  const totalPages = Math.ceil(total / limit);

  logger.debug(
    { search, page, limit, total, subscriptionStatus },
    'User list retrieved'
  );

  return {
    users,
    pagination: {
      page,
      limit,
      total,
      totalPages,
    },
  };
}

/**
 * Get detailed user information by ID
 */
export async function getUserDetail(
  userId: string
): Promise<UserDetailResponse | null> {
  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      email: true,
      name: true,
      createdAt: true,
      updatedAt: true,
      profilePicture: true,
      goalCalories: true,
      goalProtein: true,
      goalCarbs: true,
      goalFat: true,
      currentWeight: true,
      goalWeight: true,
      height: true,
      activityLevel: true,
      subscriptionTier: true,
      subscriptionBillingCycle: true,
      subscriptionStartDate: true,
      subscriptionEndDate: true,
      subscriptionPrice: true,
      appleId: true,
      _count: {
        select: {
          meals: true,
          healthMetrics: true,
          activities: true,
          supplements: true,
        },
      },
    },
  });

  if (!user) {
    return null;
  }

  // Get recent activity counts
  const [mealsLast7Days, healthMetricsLast7Days, activitiesLast7Days] =
    await Promise.all([
      prisma.meal.count({
        where: {
          userId,
          createdAt: { gte: sevenDaysAgo },
        },
      }),
      prisma.healthMetric.count({
        where: {
          userId,
          createdAt: { gte: sevenDaysAgo },
        },
      }),
      prisma.activity.count({
        where: {
          userId,
          createdAt: { gte: sevenDaysAgo },
        },
      }),
    ]);

  return {
    ...user,
    recentActivity: {
      mealsLast7Days,
      healthMetricsLast7Days,
      activitiesLast7Days,
    },
  };
}

/**
 * Export all user data for GDPR compliance
 * Returns all user data in a structured format for download
 */
export async function exportUserData(userId: string): Promise<{
  user: Record<string, unknown>;
  meals: unknown[];
  healthMetrics: unknown[];
  activities: unknown[];
  supplements: unknown[];
  supplementLogs: unknown[];
} | null> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: {
      meals: {
        orderBy: { createdAt: 'desc' },
      },
      healthMetrics: {
        orderBy: { createdAt: 'desc' },
      },
      activities: {
        orderBy: { createdAt: 'desc' },
      },
      supplements: {
        orderBy: { createdAt: 'desc' },
      },
      supplementLogs: {
        orderBy: { createdAt: 'desc' },
      },
    },
  });

  if (!user) {
    return null;
  }

  // Remove sensitive fields
  const {
    password: _password,
    resetToken: _resetToken,
    resetTokenExpiresAt: _resetTokenExpiresAt,
    meals,
    healthMetrics,
    activities,
    supplements,
    supplementLogs,
    ...userData
  } = user;

  logger.info({ userId }, 'User data exported for GDPR');

  return {
    user: userData,
    meals,
    healthMetrics,
    activities,
    supplements,
    supplementLogs,
  };
}

/**
 * Delete user account for GDPR compliance
 * Cascades deletion to all related records
 */
export async function deleteUserAccount(
  userId: string,
  adminUserId: string,
  reason: string
): Promise<boolean> {
  // Verify user exists
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { id: true, email: true },
  });

  if (!user) {
    return false;
  }

  // Use transaction to delete all related data
  await prisma.$transaction(async (tx) => {
    // Delete related records in order (respecting foreign keys)
    await tx.supplementLog.deleteMany({ where: { userId } });
    await tx.supplement.deleteMany({ where: { userId } });
    await tx.mLInsight.deleteMany({ where: { userId } });
    await tx.mLPrediction.deleteMany({ where: { userId } });
    await tx.mLFeature.deleteMany({ where: { userId } });
    await tx.userMLProfile.deleteMany({ where: { userId } });
    await tx.activity.deleteMany({ where: { userId } });
    await tx.healthMetric.deleteMany({ where: { userId } });
    await tx.weightRecord.deleteMany({ where: { userId } });
    await tx.waterIntake.deleteMany({ where: { userId } });
    await tx.meal.deleteMany({ where: { userId } });

    // Finally delete the user
    await tx.user.delete({ where: { id: userId } });
  });

  logger.warn(
    { userId, userEmail: user.email, adminUserId, reason },
    'User account deleted (GDPR)'
  );

  return true;
}
